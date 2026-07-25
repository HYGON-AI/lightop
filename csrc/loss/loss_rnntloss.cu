#include <THC/THCDeviceUtils.cuh>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#include <ATen/native/cuda/LaunchUtils.h>
#include <c10/macros/Macros.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <type_traits>
#include <numeric>
#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <torch/extension.h>
#include <torch/autograd.h>

#if defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#else

#include <cuda_fp16.h>
#include <cuda.h>

#endif

#include "loss_rnntloss.cuh"

namespace at {
  namespace native {

    namespace rnntloss_tools {

      // kernel: get max, from torchaudio
      template<int nthreads, typename DTYPE, typename CAST_DTYPE>
      __global__ void ReduceMax2D(
        int dim,
        const DTYPE *inputs, // [N, dim]
        CAST_DTYPE *outputs) {

          __shared__ CAST_DTYPE shared[nthreads];

          // each thread reduces one matrix row
          int offset = blockIdx.x * dim; // [n, 0]
          CAST_DTYPE val = CAST_DTYPE(inputs[offset]); // default = inputs(n, 0)
          for (int d = threadIdx.x; d < dim; d += nthreads) {
              CAST_DTYPE next = CAST_DTYPE(inputs[offset + d]);
              if (next > val) {
                  val = next;
              }
          }

          shared[threadIdx.x] = val;
          __syncthreads();

          for (int stride = (nthreads >> 1); stride >= WARP_SIZE; stride >>= 1) {
              if (threadIdx.x < stride && threadIdx.x + stride < dim) {
                  if (shared[threadIdx.x + stride] > shared[threadIdx.x]) {
                      shared[threadIdx.x] = shared[threadIdx.x + stride];
                      val = shared[threadIdx.x];
                  }
              }
              __syncthreads();
          }

          CAST_DTYPE shf;
          for (int stride = (WARP_SIZE >> 1); stride > 0; stride >>= 1) {
              shf = WARP_SHFL_DOWN(val, stride);
              if (threadIdx.x < stride && threadIdx.x + stride < dim) {
                  if (shf > val) {
                      val = shf;
                  }
              }
          }

          if (threadIdx.x == 0) {
              outputs[blockIdx.x] = val;
          }
      }

      template<int nthreads, typename DTYPE, typename CAST_DTYPE>
      __global__ void ReduceLogSumExpGivenMax2D(
        int dim,
        const DTYPE *inputs, // [N, dim]
        CAST_DTYPE *outputs) { // in: max -> out: logsum

          __shared__ CAST_DTYPE shared[nthreads];

          CAST_DTYPE max = outputs[blockIdx.x];
          CAST_DTYPE val = 0;

          int offset = blockIdx.x * dim;
          for (int d = threadIdx.x; d < dim; d += nthreads) {
              val += c10::cuda::compat::exp(CAST_DTYPE(inputs[offset + d]) - max);
          }

          shared[threadIdx.x] = val;
          __syncthreads();

          for (int stride = (nthreads >> 1); stride >= WARP_SIZE; stride >>= 1) {
              if (threadIdx.x < stride && threadIdx.x + stride < dim) {
                  val = shared[threadIdx.x] + shared[threadIdx.x + stride];
                  shared[threadIdx.x] = val;
              }
              __syncthreads();
          }

          CAST_DTYPE shf;
          for (int stride = (WARP_SIZE >> 1); stride > 0; stride >>= 1) {
              shf = WARP_SHFL_DOWN(val, stride);
              if (threadIdx.x < stride && threadIdx.x + stride < dim) {
                  val = val + shf;
              }
          }

          if (threadIdx.x == 0) {
              outputs[blockIdx.x] = -max - c10::cuda::compat::log(val);  // logits - max - log(sum(exp(logits-max)))
          }
      }

      template<int nthreads_for_log, typename DTYPE, typename CAST_DTYPE>
      void LogSumExp2D(
        const cudaStream_t &stream,
        const int &sizeBTU,
        const int &numTgt,
        const DTYPE *logits, //  input logits, size:[N, dim]
        CAST_DTYPE *outputs) {

          { // compute max among numTgt.
              dim3 block_dims(sizeBTU);
              dim3 thread_dims(nthreads_for_log);
              ReduceMax2D<nthreads_for_log, DTYPE, CAST_DTYPE>
              <<<block_dims, thread_dims, 0, stream>>>(
                /*dim=*/numTgt,
                /*inputs=*/logits,
                /*outputs=*/outputs);

              if (cudaGetLastError() != cudaSuccess) {
                  std::string msg("CUDA error - compute denominator reduce max failed: ");
                  throw std::runtime_error(msg);
              }
          }

          { // compute log(sum(exp(d_i - max)))
              dim3 block_dims(sizeBTU);
              dim3 thread_dims(nthreads_for_log);
              ReduceLogSumExpGivenMax2D<nthreads_for_log, DTYPE, CAST_DTYPE>
              <<<block_dims, thread_dims, 0, stream>>>(
                /*dim=*/numTgt,
                /*inputs=*/logits,
                /*outputs=*/outputs);

              if (cudaGetLastError() != cudaSuccess) {
                  std::string msg("CUDA error - compute denominator reduce sum failed: ");
                  throw std::runtime_error(msg);
              }
          }
      }

      template<typename DTYPE, typename CAST_DTYPE>
      void LogSumExp2DSelectThreadNumber(
        const cudaStream_t &stream,
        const int &sizeBTU,
        const int &numTgt,
        const DTYPE *logits, //  input logits, size:[N, dim]
        CAST_DTYPE *pDenominators) {
          if (sizeBTU <= 2048) {
              LogSumExp2D<256, DTYPE, CAST_DTYPE>(
                /*stream=*/stream,
                /*sizeBTU=*/sizeBTU,
                /*numTgt=*/numTgt,
                /*logits=*/logits, // size is (B,T,U,D)
                /*denominators=*/pDenominators);
          } else if (sizeBTU <= 4096) {
              LogSumExp2D<512, DTYPE, CAST_DTYPE>(stream, sizeBTU, numTgt, logits, pDenominators);
          } else {
              LogSumExp2D<1024, DTYPE, CAST_DTYPE>(stream, sizeBTU, numTgt, logits, pDenominators);
          }
      }

      template<typename DTYPE, typename CAST_DTYPE>
      __global__ void ComputeAlphasBetasCosts(
        int batchSize,
        int maxSrcLen,
        int maxTgtLen,
        int numTargets,
        int blankIdx,
        const DTYPE *logits, // Size: {batchSize, maxSrcLen, maxTgtLen, numTargets}
        const int *targets,
        const int *srcLengths,
        const int *tgtLengths,
        const CAST_DTYPE *denominators,
        CAST_DTYPE *alphas, // Size: {batchSize, maxSrcLen, maxTgtLen}
        CAST_DTYPE *betas, // Size: {batchSize, maxSrcLen, maxTgtLen}
        DTYPE *costs,
        const bool compute_mean) {

          const int batch = blockIdx.y;
          const int txt_id = threadIdx.x + blockIdx.z * numTargets;
          const int srcLen = srcLengths[batch];
          const int tgtLen = tgtLengths[batch] + 1; // Note that start of the sentence is added as 1 here
          if (txt_id >= tgtLen) { return; }

          // get tgt
          const int *target = targets + batch * (maxTgtLen - 1);
          // get logProb
          const int batchOffset = batch * maxSrcLen * maxTgtLen;
          const DTYPE *logit = logits + batchOffset * numTargets;
          const CAST_DTYPE *denom = denominators + batchOffset;

          // start txt id
          int u = txt_id;
          Indexer2D idxer2d(maxTgtLen);
          Indexer3D idxer3d(maxTgtLen, numTargets);

          if (blockIdx.x == 0) { // compute alpha
              // alpha path
              CAST_DTYPE *batchAlpha = alphas + batchOffset;
              if (u == 0)
                  batchAlpha[0] = 0;
              __syncthreads();

              for (int step = 1; step < srcLen + tgtLen - 1; ++step) {
                  // Move along the diagonal wavefront to leverage available parallelism
                  for (u = txt_id; u < tgtLen; u += blockDim.x) {
                      int t = step - u;
                      if (t >= 0 and t < srcLen and u >= 0 and u < tgtLen) {
                          // Eq(16) in [1]
                          if (u == 0) {
                              // alpha(t, u) = alpha(t-1, u) * null(t-1, u)
                              batchAlpha[idxer2d(t, u)] = batchAlpha[idxer2d(t - 1, u)] + CAST_DTYPE(logit[idxer3d(t - 1, u, blankIdx)]) + denom[idxer2d(t - 1, u)];
                          } else if (t == 0) {
                              // alpha(t, u-1) = alpha(t, u-1) * y(t, u-1)
                              batchAlpha[idxer2d(t, u)] = batchAlpha[idxer2d(t, u - 1)] + CAST_DTYPE(logit[idxer3d(t, u - 1, target[u - 1])]) + denom[idxer2d(t, u - 1)];
                          } else {
                              // alpha(t, u) = alpha(t-1, u) * null(t-1, u) + alpha(t, u-1) * y(t, u-1)
                              CAST_DTYPE current = batchAlpha[idxer2d(t - 1, u)] + CAST_DTYPE(logit[idxer3d(t - 1, u, blankIdx)]) + denom[idxer2d(t - 1, u)];
                              CAST_DTYPE next = batchAlpha[idxer2d(t, u - 1)] + CAST_DTYPE(logit[idxer3d(t, u - 1, target[u - 1])]) + denom[idxer2d(t, u - 1)];
                              batchAlpha[idxer2d(t, u)] = logSumExp(next, current);
                          }
                      }
                  }
                  __syncthreads();
              }

          } else if (blockIdx.x == 1) { // compute beta
              // beta path
              CAST_DTYPE *batchBeta = betas + batchOffset;
              if (u == 0) {
                  batchBeta[idxer2d(srcLen - 1, tgtLen - 1)] =
                    CAST_DTYPE(logit[idxer3d(srcLen - 1, tgtLen - 1, blankIdx)]) + denom[idxer2d(srcLen - 1, tgtLen - 1)];
              }
              __syncthreads();

              for (int step = srcLen + tgtLen - 3; step >= 0; --step) {
                  for (u = txt_id; u < tgtLen; u += blockDim.x) {
                      int t = step - u;
                      if (t >= 0 and t < srcLen and u >= 0 and u < tgtLen) {
                          // Eq(18) in [1]
                          if (u == tgtLen - 1) {
                              // beta(t, u) = beta(t+1, u) * null(t, u)
                              batchBeta[idxer2d(t, u)] = batchBeta[idxer2d(t + 1, u)] + CAST_DTYPE(logit[idxer3d(t, u, blankIdx)]) + denom[idxer2d(t, u)];
                          } else if (t == srcLen - 1) {
                              // beta(t, u) = beta(t, u+1) * y(t, u)
                              batchBeta[idxer2d(t, u)] = batchBeta[idxer2d(t, u + 1)] + CAST_DTYPE(logit[idxer3d(t, u, target[u])]) + denom[idxer2d(t, u)];
                          } else {
                              // beta(t, u) = beta(t+1, u)*null(t, u) + beta(t, u+1)*y(t, u)
                              CAST_DTYPE current = batchBeta[idxer2d(t + 1, u)] + CAST_DTYPE(logit[idxer3d(t, u, blankIdx)]) + denom[idxer2d(t, u)];
                              CAST_DTYPE next = batchBeta[idxer2d(t, u + 1)] + CAST_DTYPE(logit[idxer3d(t, u, target[u])]) + denom[idxer2d(t, u)];
                              batchBeta[idxer2d(t, u)] = logSumExp(next, current);
                          }
                      }
                  }
                  __syncthreads();
              }
              if (txt_id == 0) {
                  costs[batch] = DTYPE(-batchBeta[0]);
              }
          }
      }

      template<typename DTYPE, typename CAST_DTYPE, typename VEC_DTYPE, int VF, int ROLL_SIZE = 1>
      __global__ void ComputeGradientsVec(
        int batchSize,
        int maxSrcLen,
        int maxTgtLen,
        int numTargets,
        int blankIdx,
        CAST_DTYPE clamp,
        const DTYPE *logits,
        const int *targets,
        const int *srcLengths,
        const int *tgtLengths,
        const CAST_DTYPE *denominators,
        const CAST_DTYPE *alphas,
        const CAST_DTYPE *betas,
        DTYPE *gradients,
        const bool compute_mean) {

          const int tid = threadIdx.x; // start tgt
          const int u = blockIdx.x; // block_dims(maxTgtLen, maxSrcLen, batchSize), 0 ~ maxTgtLen
          const int t = blockIdx.y; // 0 ~ maxSrcLen
          const int batch = blockIdx.z;
          const int srcLen = srcLengths[batch];
          const int tgtLen = tgtLengths[batch] + 1;
          const int batchOffset = batch * maxSrcLen * maxTgtLen;

          if (t >= srcLen or u >= tgtLen) { return; }

          const DTYPE *logit = logits + (batchOffset + t * maxTgtLen + u) * numTargets;

          const CAST_DTYPE *batchAlpha = alphas + batchOffset;
          const CAST_DTYPE *batchBeta = betas + batchOffset;
          const CAST_DTYPE *denom = denominators + batchOffset;
          const int *target = targets + batch * (maxTgtLen - 1);

          DTYPE *batchGrad = gradients + (batchOffset + t * maxTgtLen + u) * numTargets;
          // indexer tools
          Indexer2D idxer2d(maxTgtLen);

          __shared__ CAST_DTYPE commonFactor, theBetaTU, theBetaTUp1, theBetaTp1U;
          __shared__ int theLabelShared;

          // load and store shared variables in SMEM
          if (tid == 0) {
              commonFactor = batchAlpha[idxer2d(t, u)] + denom[idxer2d(t, u)] - batchBeta[0]; // cost = -batchBeta[0]
              theBetaTU = batchBeta[idxer2d(t, u)];

              if (t != srcLen - 1) {
                  theBetaTp1U = batchBeta[idxer2d(t + 1, u)];
              }
              if (u != tgtLen - 1) {
                  theBetaTUp1 = batchBeta[idxer2d(t, u + 1)];
                  theLabelShared = target[u];
              }
          }
          __syncthreads();

          CAST_DTYPE grad, resGrad;   // default type is float
          if (VF > 1) {
              // Variabels for vectorization
              DTYPE logitBuffer[VF], gradBuffer[VF];
              auto logitVec = reinterpret_cast<VEC_DTYPE const *>(logit);   // using 64bit type
              auto gradVec = reinterpret_cast<VEC_DTYPE *>(batchGrad);
              auto logitBufferVec = reinterpret_cast<VEC_DTYPE *>(logitBuffer);
              auto gradBufferVec = reinterpret_cast<VEC_DTYPE *>(gradBuffer);

              for (int d0 = tid * VF; d0 < numTargets; d0 += blockDim.x * VF) {
                  // Load logit in a vector form
                  *logitBufferVec = logitVec[d0 / VF];
                  // Do the update for a vector of input
#pragma unroll
                  for (int i = 0; i < VF; ++i) {
                      int d = d0 + i;

                      // Do the update
                      grad = commonFactor + CAST_DTYPE(logitBuffer[i]);
                      resGrad = c10::cuda::compat::exp(grad + theBetaTU);

                      if (u < tgtLen - 1 and d == theLabelShared) {
                          resGrad -= c10::cuda::compat::exp(grad + theBetaTUp1);
                      } else if (d == blankIdx) {
                          if (t == srcLen - 1 and u == tgtLen - 1)
                              resGrad -= c10::cuda::compat::exp(grad);
                          else if (t < srcLen - 1)
                              resGrad -= c10::cuda::compat::exp(grad + theBetaTp1U);
                      }

                      // compute mean and clamp
                      if (compute_mean) {
                          resGrad /= CAST_DTYPE(batchSize);
                      }
                      if (clamp > 0) {
                          resGrad = resGrad < clamp ? resGrad : clamp;
                          resGrad = resGrad > (-clamp) ? resGrad : (-clamp);
                      }

                      gradBuffer[i] = DTYPE(resGrad);
                  }

                  // Store gradVec in a vector form
                  gradVec[d0 / VF] = *gradBufferVec;
              }
          } else {
              for (int d0 = tid * ROLL_SIZE; d0 < numTargets; d0 += blockDim.x * ROLL_SIZE) {
#pragma unroll
                  for (int i = 0; i < ROLL_SIZE; ++i) {
                      int d = d0 + i;
                      if (d >= numTargets) { break; }

                      // Do the update
                      grad = commonFactor + CAST_DTYPE(logit[d]);
                      resGrad = c10::cuda::compat::exp(grad + theBetaTU);

                      if (u < tgtLen - 1 and d == theLabelShared) {
                          resGrad -= c10::cuda::compat::exp(grad + theBetaTUp1);
                      } else if (d == blankIdx) {
                          if (t == srcLen - 1 and u == tgtLen - 1)
                              resGrad -= c10::cuda::compat::exp(grad);
                          else if (t < srcLen - 1)
                              resGrad -= c10::cuda::compat::exp(grad + theBetaTp1U);
                      }

                      if (compute_mean) {
                          resGrad /= CAST_DTYPE(batchSize);
                      }
                      if (clamp > 0) {
                          resGrad = resGrad < clamp ? resGrad : clamp;
                          resGrad = resGrad > (-clamp) ? resGrad : (-clamp);
                      }

                      batchGrad[d] = DTYPE(resGrad);
                  }
              }
          }
      }

      template<int nthreads, typename DTYPE, typename CAST_DTYPE>
      __global__ void ComputeCostsFinal(
        int batchSize,
        const DTYPE *costs,
        DTYPE *costs_sum,
        const bool compute_sum,
        const bool compute_mean) {

          __shared__ CAST_DTYPE shared[nthreads];

          // each thread reduces one matrix row
          CAST_DTYPE val = CAST_DTYPE(0); // default = inputs(n, 0)
          for (int d = threadIdx.x; d < batchSize; d += nthreads) {
              CAST_DTYPE next = CAST_DTYPE(costs[d]);
              val += next;
          }

          shared[threadIdx.x] = val;
          __syncthreads();

          for (int stride = (nthreads >> 1); stride >= WARP_SIZE; stride >>= 1) {
              if (threadIdx.x < stride && threadIdx.x + stride < batchSize) {
                  shared[threadIdx.x] += shared[threadIdx.x + stride];
                  val = shared[threadIdx.x];
              }
              __syncthreads();
          }

          CAST_DTYPE shf;
          for (int stride = (WARP_SIZE >> 1); stride > 0; stride >>= 1) {
              shf = WARP_SHFL_DOWN(val, stride);
              if (threadIdx.x < stride && threadIdx.x + stride < batchSize) {
                  val += shf;
              }
          }

          if (threadIdx.x == 0) {
              if (compute_mean) {
                  costs_sum[0] = DTYPE(val / batchSize);
              } else {
                  costs_sum[0] = DTYPE(val);
              }
          }
      }

      //=============================================================================================//
      /* check length of srcLengths and tgtLengths */
      void LengthsCheck(RnntWorkspaceTools *wkts, const int &expect_target_length, Tensor &logits,
                        const int *srcLengths, const int *tgtLengths) {
          int maxSrcLengths = 0, maxTgtLengths = 0;
          for (int b = 0; b < wkts->GetBatchSize(); b++) {
              if (maxSrcLengths < srcLengths[b]) maxSrcLengths = srcLengths[b];
              if (maxTgtLengths < tgtLengths[b]) maxTgtLengths = tgtLengths[b];
          }
          TORCH_CHECK(wkts->GetMaxSrcLen() == maxSrcLengths, "input length mismatch");
          TORCH_CHECK(wkts->GetMaxTgtLen() == (maxTgtLengths + 1), "output length mismatch");
          TORCH_CHECK(expect_target_length == maxTgtLengths, "target length mismatch");
      }

      template<typename DTYPE, typename CAST_DTYPE, typename VEC_DTYPE>
      void ComputeRnntLoss(const DTYPE *logits, const int *targets, const int *srcLengths, const int *tgtLengths,
                           CAST_DTYPE *floatWorkspace, DTYPE *costs, DTYPE *costs_sum, DTYPE *gradients,
                           RnntWorkspaceTools *wkts, const bool &inputMemAlign,
                           const bool &compute_sum, const bool &compute_mean, const bool &already_log_softmax) {
          const cudaStream_t &stream = wkts->GetCudaStream();
          const int &batchSize = wkts->GetBatchSize();
          const int &maxSrcLen = wkts->GetMaxSrcLen();
          const int &maxTgtLen = wkts->GetMaxTgtLen();
          const int &numTgt = wkts->GetNumTargets();
          const int &blank = wkts->GetBlankIdx();
          const int &clamp = wkts->GetClamp();

          CAST_DTYPE *pDenominators = floatWorkspace + wkts->GetPointerToDenominatorsOffset();

          if (!already_log_softmax) {
              // compute denominators.
              LogSumExp2DSelectThreadNumber<DTYPE, CAST_DTYPE>(
                /*stream=*/stream,
                /*sizeBTU=*/wkts->GetDenominatorsSize(),
                /*numTgt=*/numTgt,
                /*logits=*/logits, // size is (B,T,U,D)
                /*denominators=*/pDenominators
              );
          }

          CAST_DTYPE *pAlphas = floatWorkspace + wkts->GetPointerToAlphasOffset();
          CAST_DTYPE *pBetas = floatWorkspace + wkts->GetPointerToBetasOffset();
          { // compute alphas, betas and costs.
              const int tgtPerThread = (maxTgtLen + WORK_NUM_THREADS - 1) / WORK_NUM_THREADS;
              const int tgtLenBlocks = (maxTgtLen + MaxThreadPerElement - 1) / MaxThreadPerElement;
              const dim3 block_dims(2, batchSize, tgtLenBlocks);
              const dim3 thread_dims = (std::min(MaxThreadPerElement, tgtPerThread));;

              ComputeAlphasBetasCosts<DTYPE, CAST_DTYPE>
              <<<block_dims, thread_dims, 0, stream>>>(
                /*batchSize=*/batchSize,
                /*max_src_len=*/maxSrcLen,
                /*max_tgt_len=*/maxTgtLen,
                /*num_targets=*/numTgt,
                /*blank=*/blank,
                /*logits=*/logits,
                /*targets=*/targets,
                /*srcLengths=*/srcLengths,
                /*tgtLengths=*/tgtLengths,
                /*denominators=*/pDenominators,
                /*alphas=*/pAlphas,
                /*betas=*/pBetas,
                /*costs=*/costs,
                /*compute_mean=*/compute_mean);

              if (cudaGetLastError() != cudaSuccess) {
                  std::string msg("ComputeAlphasBetasCosts failed: ");
                  throw std::runtime_error(msg);
              }
          }

          { // compute gradients.
              // Would like each thread to work on 4 hidden units
              const int tgtPerThread = (numTgt + WORK_NUM_THREADS - 1) / WORK_NUM_THREADS;   // numTgt divide WORK_NUM_THREADS
              const dim3 thread_dims(std::min(MaxThreadPerElement, std::max(WARP_SIZE, tgtPerThread))); // each thread deal a tgt vector
              const dim3 block_dims(maxTgtLen, maxSrcLen, batchSize); // blocks deal vectors

              const int vecFactor = sizeof(VEC_DTYPE) / sizeof(DTYPE);
              bool typeAlign = numTgt % vecFactor == 0;

              if (typeAlign and inputMemAlign) {
                  ComputeGradientsVec<DTYPE, CAST_DTYPE, VEC_DTYPE, vecFactor>
                  <<<block_dims, thread_dims, 0, stream>>>(
                    /*batchSize=*/batchSize,
                    /*max_src_len=*/maxSrcLen,
                    /*max_tgt_len=*/maxTgtLen,
                    /*num_targets=*/numTgt,
                    /*blank=*/blank,
                    /*clamp=*/clamp,
                    /*logits=*/logits,
                    /*targets=*/targets,
                    /*srcLengths=*/srcLengths,
                    /*tgtLengths=*/tgtLengths,
                    /*denominators=*/pDenominators,
                    /*alphas=*/pAlphas,
                    /*betas=*/pBetas,
                    /*gradients=*/gradients,
                    /*compute_mean=*/compute_mean);
              } else {
                  ComputeGradientsVec<DTYPE, CAST_DTYPE, DTYPE, 1, /*ROLL_SIZE=*/4>
                  <<<block_dims, thread_dims, 0, stream>>>(
                    /*batchSize=*/batchSize,
                    /*max_src_len=*/maxSrcLen,
                    /*max_tgt_len=*/maxTgtLen,
                    /*num_targets=*/numTgt,
                    /*blank=*/blank,
                    /*clamp=*/clamp,
                    /*logits=*/logits,
                    /*targets=*/targets,
                    /*srcLengths=*/srcLengths,
                    /*tgtLengths=*/tgtLengths,
                    /*denominators=*/pDenominators,
                    /*alphas=*/pAlphas,
                    /*betas=*/pBetas,
                    /*gradients=*/gradients,
                    /*compute_mean=*/compute_mean);
              }

              if (cudaGetLastError() != cudaSuccess) {
                  std::string msg("ComputeGradients failed: ");
                  throw std::runtime_error(msg);
              }
          }

          if (compute_sum or compute_mean) { // compute final cost.
              const dim3 block_dims(1);
              const dim3 thread_dims(REDUCE_THREADS);
              ComputeCostsFinal<REDUCE_THREADS, DTYPE, CAST_DTYPE>
              <<<block_dims, thread_dims, 0, stream>>>(
                /*batchSize=*/batchSize,
                /*costs=*/costs,
                /*costs_sum=*/costs_sum,
                /*compute_sum=*/compute_sum,
                /*compute_mean=*/compute_mean);

              if (cudaGetLastError() != cudaSuccess) {
                  std::string msg("ComputeCostsFinal failed: ");
                  throw std::runtime_error(msg);
              }
          }

          return;
      }
    }

    // Entry point into RNNT Loss
    std::tuple <Tensor, Tensor> compute_accelerate_rnnt_loss(
      Tensor &logits,
      const Tensor &targets,
      const Tensor &logit_lengths,
      const Tensor &target_lengths,
      int blank,
      double clamp,
      bool compute_sum,
      bool compute_mean,
      bool check_lengths) {

        /* Check and prepare paramters */
        TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor")
        TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
        TORCH_CHECK(logit_lengths.is_cuda(), "logit_lengths must be a CUDA tensor")
        TORCH_CHECK(logit_lengths.is_contiguous(), "logit_lengths must be contiguous");
        TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor")
        TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
        TORCH_CHECK(target_lengths.is_cuda(), "target_lengths must be a CUDA tensor")
        TORCH_CHECK(target_lengths.is_contiguous(), "target_lengths must be contiguous");

        TORCH_CHECK(logits.dtype() == at::kFloat || logits.dtype() == at::kHalf,
                    "logits must be float32 or float16 (half) type");
        TORCH_CHECK(targets.dtype() == at::kInt, "targets must be int32 type");
        TORCH_CHECK(logit_lengths.dtype() == at::kInt, "logit_lengths must be int32 type");
        TORCH_CHECK(target_lengths.dtype() == at::kInt, "target_lengths must be int32 type");
        TORCH_CHECK(logits.dim() == 4, "logits must be 4-D (batch, time, target, class)");
        TORCH_CHECK(targets.dim() == 2, "targets must be 2-D (batch, max target length)");
        TORCH_CHECK(logit_lengths.dim() == 1, "logit_lengths must be 1-D");
        TORCH_CHECK(target_lengths.dim() == 1, "target_lengths must be 1-D");

        TORCH_CHECK(logit_lengths.size(0) == logits.size(0), "batch dimension mismatch between logits and logit_lengths");
        TORCH_CHECK(target_lengths.size(0) == logits.size(0), "batch dimension mismatch between logits and target_lengths");
        TORCH_CHECK(targets.size(0) == logits.size(0), "batch dimension mismatch between logits and targets");
        TORCH_CHECK(blank >= 0 && blank < logits.size(-1), "blank must be within [0, logits.shape[-1])");

        // Set options
        rnntloss_tools::RnntWorkspaceTools wkts(
          /*logits=*/logits,
          /*clamp=*/clamp,
          /*blank=*/blank);

        wkts.SetCudaStream(at::cuda::getCurrentCUDAStream());

        if (check_lengths) {
            // use cpu is faster than cuda
            rnntloss_tools::LengthsCheck(&wkts, targets.size(1), logits,
                                         logit_lengths.data_ptr<int>(), target_lengths.data_ptr<int>());
        }

        // create workspace
        Tensor float_workspace = at::zeros(wkts.GetFloatWorkspaceSize(),
                                           at::TensorOptions().device(logits.device()).dtype(ScalarType::Float));
        // set output
        Tensor costs = at::empty(wkts.GetBatchSize(), logits.options());
        Tensor gradients = at::zeros_like(logits);
        Tensor costs_sum = at::zeros({1}, logits.options());

        // vector compute
        using VEC_DTYPE = uint64_t;
        constexpr int vecAlignment = std::alignment_of<VEC_DTYPE>::value;

        switch (logits.scalar_type()) {
            case ScalarType::Float: {
                bool inputMemAlign = reinterpret_cast<VEC_DTYPE>(logits.data_ptr<float>()) % vecAlignment == 0
                                     and reinterpret_cast<VEC_DTYPE>(gradients.data_ptr<float>()) % vecAlignment == 0;

                if (wkts.GetDenominatorsSize() > 1 << (4 + 16 - sizeof(float))) { // 2^16 is the threshold
                    Tensor logits_logsoftmax = at::log_softmax(logits, 3, logits.scalar_type());
                    rnntloss_tools::ComputeRnntLoss</*DTYPE=*/float, /*CAST_DTYPE=*/float, /*VEC_DTYPE=*/VEC_DTYPE>(
                      /*logits=*/logits_logsoftmax.data_ptr<float>(),
                      /*targets=*/targets.data_ptr<int>(),
                      /*logit_lengths=*/logit_lengths.data_ptr<int>(),
                      /*target_lengths=*/target_lengths.data_ptr<int>(),
                      /*float_workspace=*/float_workspace.data_ptr<float>(),
                      /*costs=*/costs.data_ptr<float>(),
                      /*costs_sum=*/costs_sum.data_ptr<float>(),
                      /*gradients=*/gradients.data_ptr<float>(),
                      /*RnntWorkspaceTools=*/&wkts,
                      /*inputMemAlign=*/inputMemAlign,
                      /*compute_sum=*/compute_sum,
                      /*compute_mean=*/compute_mean,
                      /*already_log_softmax=*/true);
                } else {
                    rnntloss_tools::ComputeRnntLoss</*DTYPE=*/float, /*CAST_DTYPE=*/float, /*VEC_DTYPE=*/VEC_DTYPE>(
                      /*logits=*/logits.data_ptr<float>(),
                      /*targets=*/targets.data_ptr<int>(),
                      /*logit_lengths=*/logit_lengths.data_ptr<int>(),
                      /*target_lengths=*/target_lengths.data_ptr<int>(),
                      /*float_workspace=*/float_workspace.data_ptr<float>(),
                      /*costs=*/costs.data_ptr<float>(),
                      /*costs_sum=*/costs_sum.data_ptr<float>(),
                      /*gradients=*/gradients.data_ptr<float>(),
                      /*RnntWorkspaceTools=*/&wkts,
                      /*inputMemAlign=*/inputMemAlign,
                      /*compute_sum=*/compute_sum,
                      /*compute_mean=*/compute_mean,
                      /*already_log_softmax=*/false);
                }

                break;
            }
            case ScalarType::Half: {
                bool inputMemAlign = reinterpret_cast<VEC_DTYPE>(logits.data_ptr<c10::Half>()) % vecAlignment == 0
                                     and reinterpret_cast<VEC_DTYPE>(gradients.data_ptr<c10::Half>()) % vecAlignment == 0;

                rnntloss_tools::ComputeRnntLoss</*DTYPE=*/c10::Half, /*CAST_DTYPE=*/float, /*VEC_DTYPE=*/VEC_DTYPE>(
                  /*logits=*/logits.data_ptr<c10::Half>(),
                  /*targets=*/targets.data_ptr<int>(),
                  /*logit_lengths=*/logit_lengths.data_ptr<int>(),
                  /*target_lengths=*/target_lengths.data_ptr<int>(),
                  /*float_workspace=*/float_workspace.data_ptr<float>(),
                  /*costs=*/costs.data_ptr<c10::Half>(),
                  /*costs_sum=*/costs_sum.data_ptr<c10::Half>(),
                  /*gradients=*/gradients.data_ptr<c10::Half>(),
                  /*RnntWorkspaceTools=*/&wkts,
                  /*inputMemAlign=*/inputMemAlign,
                  /*compute_sum=*/compute_sum,
                  /*compute_mean=*/compute_mean,
                  /*already_log_softmax=*/false);

                break;
            }
            default: {
                break;
            }
        }

        if (compute_sum or compute_mean) {
            return std::make_tuple(costs_sum, gradients.contiguous());
        }
        return std::make_tuple(costs, gradients.contiguous());
    }

    class rnntlossFunction : public torch::autograd::Function<rnntlossFunction> {
    public:
      static torch::autograd::variable_list
      forward(torch::autograd::AutogradContext *ctx, torch::autograd::Variable &logits, const at::Tensor &targets, const at::Tensor &logit_lengths,
              const at::Tensor &target_lengths, int blank, double clamp, bool compute_sum, bool compute_mean, bool check_lengths) {
          auto result = compute_accelerate_rnnt_loss(logits, targets, logit_lengths, target_lengths, blank, clamp,
                                                     compute_sum, compute_mean, check_lengths);   // result: costs, grads
          ctx->save_for_backward({std::get<1>(result)});

          if (compute_sum or compute_mean) {
              return {std::get<0>(result)[0]};
          } else {
              return {std::get<0>(result)};
          }
      }

      static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, const torch::autograd::variable_list &gradOutput) {
          auto saved = ctx->get_saved_variables();
          auto grads = saved[0];

          return {grads, torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable(),
                  torch::autograd::Variable(), torch::autograd::Variable(),
                  torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable()};
      }
    };

    Tensor accelerate_rnnt_loss(Tensor &logits, const Tensor &targets, const Tensor &logit_lengths, const Tensor &target_lengths, int blank, double clamp,
                                bool compute_sum, bool compute_mean, bool check_lengths) {
        auto costs = rnntlossFunction::apply(logits, targets, logit_lengths, target_lengths, blank, clamp, compute_sum, compute_mean, check_lengths);
        return costs[0];
    }
  }
} // namespace at::native
