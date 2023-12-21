#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#include <ATen/native/cuda/LaunchUtils.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/macros/Macros.h>
#include <numeric>
#include <torch/autograd.h>
#include <torch/extension.h>
#include <type_traits>


#if defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#else
#include <cuda.h>
#include <cuda_fp16.h>
#endif

#include "loss_ctcloss.cuh"

namespace at {
  namespace native {

    namespace ctcloss_tools {

      // kernel: get max, from torchaudio
      template<int nthreads, typename DTYPE, typename CAST_DTYPE>
      __global__ void ReduceMax2D(int dim, const DTYPE *__restrict__ inputs,// [N, dim]
                                  CAST_DTYPE *__restrict__ outputs) {

          __shared__ CAST_DTYPE shared[nthreads];

          // each thread reduces one matrix row
          int offset = blockIdx.x * dim;              // [n, 0]
          CAST_DTYPE val = CAST_DTYPE(inputs[offset]);// default = inputs(n, 0)
          for (int d = threadIdx.x; d < dim; d += nthreads) {
              CAST_DTYPE next = CAST_DTYPE(inputs[offset + d]);
              if (next > val) { val = next; }
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
              shf = __shfl_down(val, stride);
              if (threadIdx.x < stride && threadIdx.x + stride < dim) {
                  if (shf > val) { val = shf; }
              }
          }

          if (threadIdx.x == 0) { outputs[blockIdx.x] = val; }
      }

      template<int nthreads, typename DTYPE, typename CAST_DTYPE>
      __global__ void ReduceLogSumExpGivenMax2D(int dim, const DTYPE *__restrict__ inputs,// [N, dim]
                                                CAST_DTYPE *__restrict__ outputs) {       // in: max -> out: logsum

          __shared__ CAST_DTYPE shared[nthreads];

          CAST_DTYPE max = outputs[blockIdx.x];
          CAST_DTYPE val = 0;

          int offset = blockIdx.x * dim;
          for (int d = threadIdx.x; d < dim; d += nthreads) { val += std::exp(CAST_DTYPE(inputs[offset + d]) - max); }

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
              shf = __shfl_down(val, stride);
              if (threadIdx.x < stride && threadIdx.x + stride < dim) { val = val + shf; }
          }

          if (threadIdx.x == 0) {
              outputs[blockIdx.x] = -max - std::log(val);// logits - max - log(sum(exp(logits-max)))
          }
      }

      template<int nthreads_for_log, typename DTYPE, typename CAST_DTYPE>
      void LogSumExp2D(cudaStream_t stream, int sizeBTU, int numTgt, const DTYPE *__restrict__ logits,//  input logits, size:[N, dim]
                       CAST_DTYPE *__restrict__ outputs) {

          {// compute max among numTgt.
              dim3 grid_dims(sizeBTU);
              dim3 block_dims(nthreads_for_log);
              ReduceMax2D<nthreads_for_log, DTYPE, CAST_DTYPE><<<grid_dims, block_dims, 0, stream>>>(
                /*dim=*/numTgt,
                /*inputs=*/logits,
                /*outputs=*/outputs);

              // BUGBUG: These error codes are only accurate when launching with
              // blocking. Otherwise they usually reflect earlier errors.
              if (cudaGetLastError() != cudaSuccess) {
                  std::string msg("CUDA error - compute denominator reduce max failed: ");
                  throw std::runtime_error(msg);
              }
          }

          {// compute log(sum(exp(d_i - max)))
              dim3 grid_dims(sizeBTU);
              dim3 block_dims(nthreads_for_log);
              ReduceLogSumExpGivenMax2D<nthreads_for_log, DTYPE, CAST_DTYPE><<<grid_dims, block_dims, 0, stream>>>(
                /*dim=*/numTgt,
                /*inputs=*/logits,
                /*outputs=*/outputs);

              if (cudaGetLastError() != cudaSuccess) {
                  std::string msg("CUDA error - compute denominator reduce sum failed: ");
                  throw std::runtime_error(msg);
              }
          }

          return;
      }

      template<typename DTYPE, typename CAST_DTYPE>
      void LogSumExp2DSelectThreadNumber(const cudaStream_t &stream, const int &sizeBTU, const int &numTgt,
                                         const DTYPE *__restrict__ logits,//  input logits, size:[N, dim]
                                         CAST_DTYPE *__restrict__ pDenominators) {
          if (sizeBTU <= 2048) {
              LogSumExp2D<256, DTYPE, CAST_DTYPE>(
                /*stream=*/stream,
                /*sizeBTU=*/sizeBTU,
                /*numTgt=*/numTgt,
                /*logits=*/logits,// size is (B,T,U,D)
                /*denominators=*/pDenominators);
          } else if (sizeBTU <= 4096) {
              LogSumExp2D<512, DTYPE, CAST_DTYPE>(stream, sizeBTU, numTgt, logits, pDenominators);
          } else {
              LogSumExp2D<1024, DTYPE, CAST_DTYPE>(stream, sizeBTU, numTgt, logits, pDenominators);
          }
      }

      template<typename CAST_DTYPE>
      __global__ void ComputeRepeatsAndLabelswithblank(int blankIdx, int maxTgtLen, int maxTgtWithBlankLen,
                                                       const int *__restrict__ targets,                        // Size: {batchSize, maxTgtLen}
                                                       const int *__restrict__ TgtLenCumsum,                   // targets_lengths.cumsum()
                                                       const int *__restrict__ tgtLengths,                     // Size: {batchSize}
                                                       int *__restrict__ pTargetsWithBlank,                    // Size: {batchSize, maxTgtWithBlankLen }
                                                       int *__restrict__ pTargetRepeats) {                     // Size: {batchSize}
          int uid = threadIdx.x;
          int batch = blockIdx.x;

          const int tgtLen = tgtLengths[batch];


          int tgt_step = 0;
          if(batch > 0) tgt_step = TgtLenCumsum[batch - 1];

          const int *tgtVec = targets + tgt_step;

          int *tgtWithBlankVec = pTargetsWithBlank + batch * maxTgtWithBlankLen;
          pTargetRepeats[batch] = 0;

          if (uid > maxTgtLen) return;

          for (int u = uid; u <= maxTgtLen; u += blockDim.x) {
              // set value, u <= tgtLen
              tgtWithBlankVec[2 * u] = blankIdx;
              if (u < tgtLen) {     // u <= tgtLen-1
                  tgtWithBlankVec[2 * u + 1] = tgtVec[u];

                  if (u > 0 and tgtVec[u - 1] == tgtVec[u]) {// repeat label
                      pTargetRepeats[batch]++;
                  }
              } else {
                  tgtWithBlankVec[2 * u + 1] = blankIdx;
              }
          }
      }

      template<typename DTYPE, typename CAST_DTYPE>
      __global__ void ComputeAlphasBetasCosts(const int batchSize, const int maxSrcLen, const int maxTgtLen, const int maxTgtWithBlankLen,
                                              const int numTargets, const int blankIdx,
                                              const DTYPE *__restrict__ logits,           // Size: {batchSize, maxSrcLen, numTargets}
                                              const int *__restrict__ pTargetsWithBlank,  // Size: {batchSize, maxTgtWithBlankLen}
                                              const int *__restrict__ srcLengths,         // Size: {batchSize}
                                              const int *__restrict__ tgtLengths,         // Size: {batchSize}
                                              const CAST_DTYPE *__restrict__ denominators,// Size: {batchSize, maxSrcLen}
                                              const int *__restrict__ pTargetRepeats,     // Size: {batchSize}
                                              CAST_DTYPE *__restrict__ alphas,            // Size: {batchSize, maxSrcLen, maxTgtWithBlankLen}
                                              CAST_DTYPE *__restrict__ betas,             // Size: {batchSize, maxSrcLen, maxTgtWithBlankLen}
                                              DTYPE *__restrict__ costs,
                                              const bool compute_mean, const CAST_DTYPE neginf) {

          const int uid = threadIdx.x;  // target u
          const int batch = threadIdx.y + blockIdx.y * blockDim.y;
          if (batch >= batchSize) { return; }

          const int srcLen = srcLengths[batch];
          const int tgtLen = tgtLengths[batch];// Note that start of the sentence is added as 1 here
          const int tgtWithBlankLen = 2 * tgtLen + 1;
          // get tgt
          const int *tgtWithBlankVec = pTargetsWithBlank + batch * maxTgtWithBlankLen;
          // get logProb
          const int batchOffset = batch * maxSrcLen;// NTC
          const DTYPE *logit = logits + batchOffset * numTargets;
          const CAST_DTYPE *denomVec = denominators + batchOffset;
          const int tgt_repeat = pTargetRepeats[batch];
          // alpha path
          CAST_DTYPE *batchAlpha = alphas + batchOffset * maxTgtWithBlankLen;// Size: {batchSize, maxSrcLen, maxTgtWithBlankLen}
          // beta path
          CAST_DTYPE *batchBeta = betas + batchOffset * maxTgtWithBlankLen;// Size: {batchSize, maxSrcLen, maxTgtWithBlankLen}

          // set Indexer
          Indexer2D idxer2d_src(numTargets);
          Indexer2D idxer2d_tgt(maxTgtWithBlankLen);

          if (tgtLen + tgt_repeat > srcLen) { return; }
          if (uid >= tgtWithBlankLen) { return; }

          if (blockIdx.x == 0) { // compute alpha

#pragma unroll
              for (int u = uid; u < tgtWithBlankLen; u += blockDim.x) {
                  // get log probs of label
                  if (u == 0) {
                      batchAlpha[idxer2d_tgt(0, 0)] = CAST_DTYPE(logit[idxer2d_src(0, tgtWithBlankVec[0])]) + denomVec[0];
                      batchAlpha[idxer2d_tgt(0, 1)] = CAST_DTYPE(logit[idxer2d_src(0, tgtWithBlankVec[1])]) + denomVec[0];
                  }

                  for (int t = 1; t < srcLen; ++t) {
                      __syncthreads();

                      CAST_DTYPE alpha_t_i;
                      if (u >= 1) {
                          CAST_DTYPE la1 = batchAlpha[idxer2d_tgt(t - 1, u)];
                          CAST_DTYPE la2 = batchAlpha[idxer2d_tgt(t - 1, u - 1)];
                          CAST_DTYPE lamax = std::max(la1, la2);

                          if (u >= 2 and tgtWithBlankVec[u] != blankIdx and tgtWithBlankVec[u] != tgtWithBlankVec[u - 2]) {
                              CAST_DTYPE la3 = batchAlpha[idxer2d_tgt(t - 1, u - 2)];
                              if (la3 > lamax) { lamax = la3; }

                              // when all are neginf. (then the whole thing is neginf, but we can pretend)
                              if (lamax == neginf) { alpha_t_i = neginf; }
                              else { alpha_t_i = logSumExp3Input(la1, la2, la3, lamax); }
                          } else {
                              // when all are neginf. (then the whole thing is neginf, but we can pretend)
                              if (lamax == neginf) { alpha_t_i = neginf; }
                              else { alpha_t_i = logSumExp2Input(la1, la2, lamax); }
                          }
                      } else {
                          alpha_t_i = batchAlpha[idxer2d_tgt(t - 1, u)];
                      }

                      batchAlpha[idxer2d_tgt(t, u)] = alpha_t_i + CAST_DTYPE(logit[idxer2d_src(t, tgtWithBlankVec[u])]) + denomVec[t];
                  }
              }
          } else if (blockIdx.x == 1) { // compute beta

#pragma unroll
              for (int u = uid; u < tgtWithBlankLen; u += blockDim.x) {
                  if (u == 0) {
                      batchBeta[idxer2d_tgt(srcLen - 1, tgtWithBlankLen - 1)] =
                        CAST_DTYPE(logit[idxer2d_src(srcLen - 1, tgtWithBlankVec[tgtWithBlankLen - 1])]) + denomVec[srcLen - 1];
                      batchBeta[idxer2d_tgt(srcLen - 1, tgtWithBlankLen - 2)] =
                        CAST_DTYPE(logit[idxer2d_src(srcLen - 1, tgtWithBlankVec[tgtWithBlankLen - 2])]) + denomVec[srcLen - 1];
                  }

                  for (int t = srcLen - 2; t >= 0; --t) {
                      __syncthreads();

                      CAST_DTYPE beta_t_i;
                      if (u < (tgtWithBlankLen - 1)) {
                          CAST_DTYPE lb1 = batchBeta[idxer2d_tgt(t + 1, u)];
                          CAST_DTYPE lb2 = batchBeta[idxer2d_tgt(t + 1, u + 1)];
                          CAST_DTYPE lbmax = std::max(lb1, lb2);

                          if (u < (tgtWithBlankLen - 2) and tgtWithBlankVec[u] != blankIdx and tgtWithBlankVec[u] != tgtWithBlankVec[u + 2]) {
                              CAST_DTYPE lb3 = batchBeta[idxer2d_tgt(t + 1, u + 2)];
                              if (lb3 > lbmax) lbmax = lb3;
                              // when all are neginf. (then the whole thing is neginf, but we can pretend)
                              if (lbmax == neginf) { beta_t_i = neginf; }
                              else { beta_t_i = logSumExp3Input(lb1, lb2, lb3, lbmax); }
                          } else {
                              // when all are neginf. (then the whole thing is neginf, but we can pretend)
                              if (lbmax == neginf) { beta_t_i = neginf; }
                              else { beta_t_i = logSumExp2Input(lb1, lb2, lbmax); }
                          }
                      } else {
                          beta_t_i = batchBeta[idxer2d_tgt(t + 1, u)];
                      }

                      batchBeta[idxer2d_tgt(t, u)] = beta_t_i + CAST_DTYPE(logit[idxer2d_src(t, tgtWithBlankVec[u])]) + denomVec[t];
                  }
              }
          }
          if (uid == 0) {
              // loss = beta(0,0) + beta(0,1)
              CAST_DTYPE lbmax = std::max(batchBeta[idxer2d_tgt(0, 0)], batchBeta[idxer2d_tgt(0, 1)]);
              costs[batch] = -DTYPE(logSumExp2Input(batchBeta[idxer2d_tgt(0, 0)], batchBeta[idxer2d_tgt(0, 1)], lbmax));
          }
      }

      template<typename DTYPE, typename CAST_DTYPE>
      __global__ void ComputeAlphaBetaSum(const int batchSize, const int maxSrcLen, const int maxTgtLen, const int maxTgtWithBlankLen, const int numTargets,
                                          const int *__restrict__ pTargetsWithBlank,      // Size: {batchSize, maxTgtLen}
                                          const DTYPE *__restrict__ logits,               // Size: {batchSize, maxSrcLen, numTargets}
                                          const CAST_DTYPE *__restrict__ denominators,    // Size: {batchSize, maxSrcLen}
                                          const int *__restrict__ srcLengths,             // Size: {batchSize}
                                          const int *__restrict__ tgtLengths,             // Size: {batchSize}
                                          const DTYPE *__restrict__ costs,                // Size: {batchSize}
                                          const CAST_DTYPE *__restrict__ alphas,          // Size: {batchSize, maxSrcLen, maxTgtWithBlankLen}
                                          const CAST_DTYPE *__restrict__ betas,           // Size: {batchSize, maxSrcLen, maxTgtWithBlankLen}
                                          CAST_DTYPE *__restrict__ pGradientAlphasBetas,  // Size: {batchSize, maxSrcLen, numTargets}
                                          const CAST_DTYPE neginf) {

          int u = threadIdx.x + blockIdx.x * blockDim.x;      // note, this directly indexes into targets, not targets prime!
          int batch = threadIdx.y + blockIdx.y * blockDim.y;
          int tid = blockIdx.z;

          if (batch >= batchSize) { return; }

          const int srcLen = srcLengths[batch];
          const int tgtLen = tgtLengths[batch];

          if (tid >= srcLen) { return; }

          const int tgtWithBlankLen = 2 * tgtLen + 1;
          const int batchOffset = batch * maxSrcLen;// NTD
          const CAST_DTYPE cost = (CAST_DTYPE) costs[batch];

          if (u >= tgtWithBlankLen) { return; }

          const int *tgtWithBlankVec = pTargetsWithBlank + batch * maxTgtWithBlankLen;
          int label = tgtWithBlankVec[u];

#pragma unroll
          for (int t = tid; t < srcLen; t += gridDim.z) {

              const CAST_DTYPE *batchAlpha = alphas + (batchOffset + t) * maxTgtWithBlankLen;
              const CAST_DTYPE *batchBeta = betas + (batchOffset + t) * maxTgtWithBlankLen;

              if (batchAlpha[u] == neginf || batchBeta[u] == neginf) { continue; }

              const DTYPE *logit = logits + (batchOffset + t) * numTargets;
              CAST_DTYPE logit_v = (CAST_DTYPE) logit[label];

              const CAST_DTYPE denom_value = *(denominators + batchOffset + t);
              CAST_DTYPE *pAlphasBetasVec = pGradientAlphasBetas + (batchOffset + t) * numTargets;// already initialized to zeros

              gpuAtomicAddNoReturn(&pAlphasBetasVec[label], std::exp(batchAlpha[u] + batchBeta[u] + cost - (logit_v + denom_value)));
          }
      }

      template<typename DTYPE, typename CAST_DTYPE, typename VEC_DTYPE, int VF>
      __global__ void ComputeGradientsVec(const int batchSize, const int maxSrcLen, const int numTargets,
                                          const CAST_DTYPE clamp, const DTYPE *__restrict__ logits,// Size: {batchSize, maxSrcLen, numTargets}
                                          const int *__restrict__ srcLengths,                      // Size: {batchSize}
                                          const int *__restrict__ tgtLengths,                      // Size: {batchSize}
                                          const CAST_DTYPE *__restrict__ denominators,             // Size: {batchSize, maxSrcLen}
                                          const DTYPE *__restrict__ costs,                         // Size: {batchSize}
                                          CAST_DTYPE *__restrict__ pGradientAlphasBetas,           // Size: {batchSize, maxSrcLen, numTargets}
                                          DTYPE *__restrict__ gradients,                           // Size: {batchSize, maxSrcLen, numTargets}
                                          const bool compute_mean) {                               // Size: {batchSize, maxSrcLen, numTargets}

          const int tid = threadIdx.x; // part of numTargets
          const int t = blockIdx.x;    // 0 ~ maxSrcLen
          const int batch = blockIdx.y;// 0 ~ batchSize

          const int srcLen = srcLengths[batch];
          const int tgtLen = tgtLengths[batch];
          const int tgtWithBlankLen = 2 * tgtLen + 1;

          if (tid >= numTargets) { return; }

          const int batchOffset = batch * maxSrcLen;// NTC
          const DTYPE *logit = logits + (batchOffset + t) * numTargets;
          const CAST_DTYPE denom_value = *(denominators + batchOffset + t);
          const DTYPE cost = costs[batch];
          CAST_DTYPE *pAlphasBetasVec = pGradientAlphasBetas + (batchOffset + t) * numTargets;
          DTYPE *batchGrad = gradients + (batchOffset + t) * numTargets;

          if (VF > 1) { // I/O Vectorized
              // Variabels for vectorization
              DTYPE logitBuffer[VF], gradBuffer[VF];
              auto logitVec = reinterpret_cast<VEC_DTYPE const *>(logit);// using 64bit type
              auto gradVec = reinterpret_cast<VEC_DTYPE *>(batchGrad);
              auto logitBufferVec = reinterpret_cast<VEC_DTYPE *>(logitBuffer);
              auto gradBufferVec = reinterpret_cast<VEC_DTYPE *>(gradBuffer);

              for (int d0 = tid * VF; d0 < numTargets; d0 += blockDim.x * VF) {
                  // Load logit in a vector form
                  *logitBufferVec = logitVec[d0 / VF];
#pragma unroll
                  for (int i = 0; i < VF; ++i) {
                      int d = d0 + i;
                      if (d >= numTargets) { break; }

                      // Do the update
                      CAST_DTYPE resGrad = std::exp(CAST_DTYPE(logit[d]) + denom_value) - pAlphasBetasVec[d];

                      // compute mean and clamp
                      resGrad /= CAST_DTYPE(batchSize);
                      if (compute_mean) { resGrad /= CAST_DTYPE(tgtLen); }

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
              for (int d = tid; d < numTargets; d += blockDim.x) {
                  // Do the update
                  CAST_DTYPE resGrad = std::exp(CAST_DTYPE(logit[d]) + denom_value) - pAlphasBetasVec[d];

                  // compute mean and clamp
                  resGrad /= CAST_DTYPE(batchSize);
                  if (compute_mean) { resGrad /= CAST_DTYPE(tgtLen); }

                  if (clamp > 0) {
                      resGrad = resGrad < clamp ? resGrad : clamp;
                      resGrad = resGrad > (-clamp) ? resGrad : (-clamp);
                  }

                  batchGrad[d] = DTYPE(resGrad);
              }
          }
      }

      template<int nthreads, typename DTYPE, typename CAST_DTYPE>
      __global__ void ComputeCostsFinal(int batchSize,
                                        const int *__restrict__ tgtLengths,
                                        const DTYPE *__restrict__ costs, DTYPE *__restrict__ costs_sum,
                                        const bool compute_sum, const bool compute_mean) {

          __shared__ CAST_DTYPE shared[nthreads];

          // each thread reduces one matrix row
          CAST_DTYPE val = CAST_DTYPE(0);// default = inputs(n, 0)
          for (int d = threadIdx.x; d < batchSize; d += nthreads) {
              CAST_DTYPE next = CAST_DTYPE(costs[d]);
              if (compute_mean) {
                  next /= tgtLengths[d];
              }
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
              shf = __shfl_down(val, stride);
              if (threadIdx.x < stride && threadIdx.x + stride < batchSize) { val += shf; }
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
      void LengthsCheck(CtcWorkspaceTools *wkts, const int *srcLengths, const int *tgtLengths) {
          int maxSrcLengths = 0, maxTgtLengths = 0;
          for (int b = 0; b < wkts->GetBatchSize(); ++b) {
              if (maxSrcLengths < srcLengths[b]) maxSrcLengths = srcLengths[b];
              if (maxTgtLengths < tgtLengths[b]) maxTgtLengths = tgtLengths[b];
          }
          TORCH_CHECK(wkts->GetMaxSrcLen() == maxSrcLengths, "input length mismatch");
          TORCH_CHECK(wkts->GetMaxTgtLen() == maxTgtLengths, "target length mismatch");
      }

      template<typename DTYPE, typename CAST_DTYPE, typename VEC_DTYPE>
      void ComputeCtcLoss(const DTYPE *__restrict__ logits, const int *__restrict__ targets,
                          const int *__restrict__ srcLengths, const int *__restrict__ tgtLengths,
                          int *__restrict__ intWorkspace, CAST_DTYPE *__restrict__ alphaBetaWorkspace,
                          CAST_DTYPE *__restrict__ floatWorkspace,
                          DTYPE *__restrict__ costs, DTYPE *__restrict__ costs_sum,
                          DTYPE *__restrict__ gradients, CtcWorkspaceTools *__restrict__ wkts,
                          const bool &inputMemAlign64, const bool &compute_sum, const bool &compute_mean, const bool &already_log_softmax) {

          const CAST_DTYPE neginf = -CAST_DTYPE(INFINITY);

          const cudaStream_t &stream = wkts->GetCudaStream();
          const int &batchSize = wkts->GetBatchSize();
          const int &maxSrcLen = wkts->GetMaxSrcLen();
          const int &maxTgtLen = wkts->GetMaxTgtLen();
          const int &maxTgtWithBlankLen = wkts->GetMaxTgtWithBlankLen();
          const int &numTgt = wkts->GetNumTargets();
          const int &blank = wkts->GetBlankIdx();
          const int &clamp = wkts->GetClamp();

          // =========================================== compute log_softmax =========================================== //
          CAST_DTYPE *pDenominators = floatWorkspace + wkts->GetPointerToDenominatorsOffset();
          if (!already_log_softmax) {
              // compute denominators.
              LogSumExp2DSelectThreadNumber<DTYPE, CAST_DTYPE>(
                /*stream=*/stream,
                /*sizeBTU=*/wkts->GetDenominatorsSize(),
                /*numTgt=*/numTgt,
                /*logits=*/logits,// size is (B,T,D)
                /*denominators=*/pDenominators);
          }

          // =========================================== compute labels_with_blank and target_repeat =========================================== //
          int *pTargetRepeats = intWorkspace + wkts->GetPointerToTargetRepeatsOffset();
          int *pTargetsWithBlank = intWorkspace + wkts->GetLabelsWithBlankOffset();
          {
              const int tgtPerThread = (maxTgtLen + WORK_NUM_THREADS - 1) / WORK_NUM_THREADS;
              const dim3 block_dims(std::min(MaxThreadPerElement, tgtPerThread));// each thread deal a tgt vector
              const dim3 grid_dims(batchSize);

              ComputeRepeatsAndLabelswithblank<CAST_DTYPE><<<grid_dims, block_dims, 0, stream>>>(
                /*blankIdx=*/blank,
                /*maxTgtLen=*/maxTgtLen,
                /*maxTgtWithBlankLen=*/maxTgtWithBlankLen,
                /*targets=*/targets,
                /*TgtLenCumsum=*/wkts->GetTgtLenCumsum(),
                /*tgtLengths=*/tgtLengths,
                /*pTargetsWithBlank=*/pTargetsWithBlank,
                /*pTargetRepeats=*/pTargetRepeats);

              if (cudaGetLastError() != cudaSuccess) {
                  std::string msg("ComputeRepeatsAndLabelswithblank failed: ");
                  throw std::runtime_error(msg);
              }
          }

          // =========================================== compute alphas, betas and costs. =========================================== //
          CAST_DTYPE *pAlphas = alphaBetaWorkspace + wkts->GetPointerToAlphasOffset();
          CAST_DTYPE *pBetas = alphaBetaWorkspace + wkts->GetPointerToBetasOffset();
          {
              int threads_target = MAX_BLOCK_SIZE;
              while (threads_target / 2 >= maxTgtWithBlankLen && threads_target > 1) {
                  threads_target /= 2;
              }
              int threads_batch = std::min(MAX_BLOCK_SIZE / threads_target, batchSize);
              dim3 block_dims(threads_target, threads_batch);
              dim3 grid_dims(2, (batchSize + threads_batch - 1) / threads_batch);

              ComputeAlphasBetasCosts<DTYPE, CAST_DTYPE><<<grid_dims, block_dims, 0, stream>>>(
                /*batchSize=*/batchSize,
                /*maxSrcLen=*/maxSrcLen,
                /*maxTgtLen=*/maxTgtLen,
                /*maxTgtWithBlankLen=*/maxTgtWithBlankLen,
                /*numTgt=*/numTgt,
                /*blank=*/blank,
                /*logits=*/logits,
                /*pTargetsWithBlank=*/pTargetsWithBlank,
                /*srcLengths=*/srcLengths,
                /*tgtLengths=*/tgtLengths,
                /*denominators=*/pDenominators,
                /*pTargetRepeats=*/pTargetRepeats,
                /*alphas=*/pAlphas,
                /*betas=*/pBetas,
                /*costs=*/costs,
                /*compute_mean=*/compute_mean,
                /*neginf=*/neginf);

              if (cudaGetLastError() != cudaSuccess) {
                  std::string msg("ComputeAlphasBetasCosts failed: ");
                  throw std::runtime_error(msg);
              }
          }

          // =========================================== compute alphas, betas sum for labels. =========================================== //
          CAST_DTYPE *pGradientAlphasBetas = floatWorkspace + wkts->GetPointerToGradientAlphasBetasOffset();
          {
              int threads_target = MAX_BLOCK_SIZE;
              while (threads_target / 2 >= maxTgtWithBlankLen && threads_target > 1) {
                  threads_target /= 2;
              }
              int threads_batch = std::min(MAX_BLOCK_SIZE / threads_target, batchSize);
              dim3 block_dims(threads_target, threads_batch);
              dim3 grid_dims(std::max<int>((maxTgtWithBlankLen + threads_target - 1) / threads_target, 1),
                             (batchSize + threads_batch - 1) / threads_batch,
                             4);

              ComputeAlphaBetaSum<DTYPE, CAST_DTYPE><<<grid_dims, block_dims, 0, stream>>>(
                /*batchSize=*/batchSize,
                /*maxSrcLen=*/maxSrcLen,
                /*maxTgtLen=*/maxTgtLen,
                /*maxTgtWithBlankLen=*/maxTgtWithBlankLen,
                /*num_targets=*/numTgt,
                /*pTargetsWithBlank=*/pTargetsWithBlank,
                /*logits=*/logits,
                /*denominators=*/pDenominators,
                /*srcLengths=*/srcLengths,
                /*tgtLengths=*/tgtLengths,
                /*costs=*/costs,
                /*alphas=*/pAlphas,
                /*betas=*/pBetas,
                /*pGradientAlphasBetas=*/pGradientAlphasBetas,
                /*neginf=*/neginf);

              if (cudaGetLastError() != cudaSuccess) {
                  std::string msg("ComputeAlphaBetaSum failed: ");
                  throw std::runtime_error(msg);
              }
          }

          // =========================================== compute gradients. =========================================== //
          {
              const int tgtPerThread = (numTgt + WORK_NUM_THREADS - 1) / WORK_NUM_THREADS;
              const dim3 block_dims(std::min(MaxThreadPerElement, tgtPerThread));   // each thread deal a tgt vector
              const dim3 grid_dims(maxSrcLen, batchSize);                           // blocks deal vectors

              const int vecFactor64 = sizeof(VEC_DTYPE) / sizeof(DTYPE);
              if (inputMemAlign64 && numTgt % vecFactor64 == 0) {
                  // Would like each thread to work on 4 hidden units
                  ComputeGradientsVec<DTYPE, CAST_DTYPE, VEC_DTYPE, vecFactor64><<<grid_dims, block_dims, 0, stream>>>(
                    /*batchSize=*/batchSize,
                    /*maxSrcLen=*/maxSrcLen,
                    /*num_targets=*/numTgt,
                    /*clamp=*/clamp,
                    /*logits=*/logits,
                    /*srcLengths=*/srcLengths,
                    /*tgtLengths=*/tgtLengths,
                    /*denominators=*/pDenominators,
                    /*costs=*/costs,
                    /*pGradientAlphasBetas=*/pGradientAlphasBetas,
                    /*gradients=*/gradients,
                    /*compute_mean=*/compute_mean);
              } else {
                  ComputeGradientsVec<DTYPE, CAST_DTYPE, DTYPE, 1><<<grid_dims, block_dims, 0, stream>>>(
                    /*batchSize=*/batchSize,
                    /*maxSrcLen=*/maxSrcLen,
                    /*num_targets=*/numTgt,
                    /*clamp=*/clamp,
                    /*logits=*/logits,
                    /*srcLengths=*/srcLengths,
                    /*tgtLengths=*/tgtLengths,
                    /*denominators=*/pDenominators,
                    /*costs=*/costs,
                    /*pGradientAlphasBetas=*/pGradientAlphasBetas,
                    /*gradients=*/gradients,
                    /*compute_mean=*/compute_mean);
              }

              if (cudaGetLastError() != cudaSuccess) {
                  std::string msg("ComputeGradients failed: ");
                  throw std::runtime_error(msg);
              }
          }

          if (compute_sum or compute_mean) {// compute final cost.
              const dim3 grid_dims(1);
              const dim3 block_dims(REDUCE_THREADS);
              ComputeCostsFinal<REDUCE_THREADS, DTYPE, CAST_DTYPE><<<grid_dims, block_dims, 0, stream>>>(
                /*batchSize=*/batchSize,
                /*tgtLengths=*/tgtLengths,
                /*costs=*/costs,
                /*costs_sum=*/costs_sum,
                /*compute_sum=*/compute_sum,
                /*compute_mean=*/compute_mean);

              if (cudaGetLastError() != cudaSuccess) {
                  std::string msg("ComputeCostsFinal failed: ");
                  throw std::runtime_error(msg);
              }
          }
      }
    }

    // Entry point into CTC Loss
    std::tuple <Tensor, Tensor> compute_accelerate_ctc_loss(Tensor &logits, const Tensor &targets, const Tensor &logit_lengths, const Tensor &target_lengths, int blank,
                                                            double clamp, bool logits_time_major, bool compute_sum, bool compute_mean, bool check_lengths) {

        /* Check and prepare paramters */
        TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor")
        TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
        TORCH_CHECK(logit_lengths.is_cuda(), "logit_lengths must be a CUDA tensor")
        TORCH_CHECK(logit_lengths.is_contiguous(), "logit_lengths must be contiguous");
        TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor")
        TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
        TORCH_CHECK(target_lengths.is_cuda(), "target_lengths must be a CUDA tensor")
        TORCH_CHECK(target_lengths.is_contiguous(), "target_lengths must be contiguous");

        TORCH_CHECK(logits.dtype() == at::kFloat || logits.dtype() == at::kHalf, "logits must be float32 or float16 (half) type");
        TORCH_CHECK(targets.dtype() == at::kInt || targets.dtype() == at::kLong, "targets must be int32 type");
        TORCH_CHECK(logit_lengths.dtype() == at::kInt || logit_lengths.dtype() == at::kLong, "logit_lengths must be int32 type");
        TORCH_CHECK(target_lengths.dtype() == at::kInt || target_lengths.dtype() == at::kLong, "target_lengths must be int32 type");
        TORCH_CHECK(logits.dim() == 3, "logits must be 3-D (time, batch, class)");        // Tensor of size (T, N, C) or (N,T,C)
        TORCH_CHECK(targets.dim() == 2 || targets.dim() == 1, "targets must be 2-D (batch, max(target_lengths)) or 1-D sum(target_lengths)");   // Tensor of size (T, S)
        // It represent the target sequences. Each element in the target sequence is a class index. And the target index cannot be blank (default=0).
        // In the (N, S) form, targets are padded to the length of the longest sequence, and stacked.
        TORCH_CHECK(logit_lengths.dim() == 1, "logit_lengths must be 1-D");
        TORCH_CHECK(target_lengths.dim() == 1, "target_lengths must be 1-D");

        if (logits_time_major) {                          // input logit size (T,N,C)
            logits = logits.permute({1, 0, 2}).contiguous();// (T,N,C) -> (N,T,C)
        }

        TORCH_CHECK(logit_lengths.size(0) == logits.size(0), "batch dimension mismatch between logits and logit_lengths");
        TORCH_CHECK(target_lengths.size(0) == logits.size(0), "batch dimension mismatch between logits and target_lengths");
        TORCH_CHECK(targets.size(0) == logits.size(0) || targets.dim() == 1, "batch dimension mismatch between logits and targets if targets ndim=2");
        TORCH_CHECK(blank >= 0 && blank < logits.size(-1), "blank must be within [0, logits.shape[-1])");

        if(targets.dtype() == at::kLong) { targets.to(at::kInt); }
        if(logit_lengths.dtype() == at::kLong) {  logit_lengths.to(at::kInt); }
        if(target_lengths.dtype() == at::kLong) { target_lengths.to(at::kInt); }

        // Set options
        ctcloss_tools::CtcWorkspaceTools wkts(
          /*logits=*/logits,
          /*targets=*/targets,
          /*target_lengths=*/target_lengths,
          /*clamp=*/clamp,
          /*blank=*/blank);

        wkts.SetCudaStream(at::cuda::getCurrentCUDAStream());

        if (check_lengths) {
            // use cpu is faster than cuda
            ctcloss_tools::LengthsCheck(&wkts, logit_lengths.data_ptr<int>(), target_lengths.data_ptr<int>());
        }

        // create int workspace
        Tensor int_workspace = at::empty(wkts.GetIntWorkspaceSize(), at::TensorOptions().device(logits.device()).dtype(ScalarType::Int));
        // create float workspace
        Tensor alpha_beta_workspace = at::empty(wkts.GetAlphaBetaWorkspaceSize(), at::TensorOptions().device(logits.device()).dtype(ScalarType::Float),
                                                LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        alpha_beta_workspace.fill_(float(-INFINITY));

        Tensor float_workspace = at::zeros(wkts.GetFloatWorkspaceSize(), at::TensorOptions().device(logits.device()).dtype(ScalarType::Float));

        Tensor gradients = at::zeros_like(logits, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

        // set output
        Tensor costs = at::empty(wkts.GetBatchSize(), logits.options());
        Tensor costs_sum = at::zeros({1}, logits.options());

        // vector compute
        using VEC_DTYPE = uint64_t;
        constexpr auto vecAlignment_64 = std::alignment_of<VEC_DTYPE>::value;

        switch (logits.scalar_type()) {
            case ScalarType::Float: {

                auto p_logits_64 = reinterpret_cast<VEC_DTYPE>(logits.data_ptr<float>());
                auto p_gradients_64 = reinterpret_cast<VEC_DTYPE>(gradients.data_ptr<float>());
                bool inputMemAlign64 = p_logits_64 % vecAlignment_64 == 0 && p_gradients_64 % vecAlignment_64 == 0;

                if (wkts.GetDenominatorsSize() > 1 << (16 - sizeof(float))) {// 2^12 is the threshold
                    Tensor logits_logsoftmax = at::log_softmax(logits, 2, logits.scalar_type());

                    ctcloss_tools::ComputeCtcLoss</*DTYPE=*/float, /*CAST_DTYPE=*/float, /*VEC_DTYPE=*/VEC_DTYPE>(
                      /*logits=*/logits_logsoftmax.data_ptr<float>(),
                      /*targets=*/targets.data_ptr<int>(),
                      /*logit_lengths=*/logit_lengths.data_ptr<int>(),
                      /*target_lengths=*/target_lengths.data_ptr<int>(),
                      /*int_workspace=*/int_workspace.data_ptr<int>(),
                      /*alpha_beta_workspace=*/alpha_beta_workspace.data_ptr<float>(),
                      /*float_workspace=*/float_workspace.data_ptr<float>(),
                      /*costs=*/costs.data_ptr<float>(),
                      /*costs_sum=*/costs_sum.data_ptr<float>(),
                      /*gradients=*/gradients.data_ptr<float>(),
                      /*CtcWorkspaceTools=*/&wkts,
                      /*inputMemAlign64=*/inputMemAlign64,
                      /*compute_sum=*/compute_sum,
                      /*compute_mean=*/compute_mean,
                      /*already_log_softmax=*/true);
                } else {
                    ctcloss_tools::ComputeCtcLoss</*DTYPE=*/float, /*CAST_DTYPE=*/float, /*VEC_DTYPE=*/VEC_DTYPE>(
                      /*logits=*/logits.data_ptr<float>(),
                      /*targets=*/targets.data_ptr<int>(),
                      /*logit_lengths=*/logit_lengths.data_ptr<int>(),
                      /*target_lengths=*/target_lengths.data_ptr<int>(),
                      /*int_workspace=*/int_workspace.data_ptr<int>(),
                      /*alpha_beta_workspace=*/alpha_beta_workspace.data_ptr<float>(),
                      /*float_workspace=*/float_workspace.data_ptr<float>(),
                      /*costs=*/costs.data_ptr<float>(),
                      /*costs_sum=*/costs_sum.data_ptr<float>(),
                      /*gradients=*/gradients.data_ptr<float>(),
                      /*CtcWorkspaceTools=*/&wkts,
                      /*inputMemAlign64=*/inputMemAlign64,
                      /*compute_sum=*/compute_sum,
                      /*compute_mean=*/compute_mean,
                      /*already_log_softmax=*/false);
                }
                break;
            }

            case ScalarType::Half: {
                auto p_logits_64 = reinterpret_cast<VEC_DTYPE>(logits.data_ptr<c10::Half>());
                auto p_gradients_64 = reinterpret_cast<VEC_DTYPE>(gradients.data_ptr<c10::Half>());
                bool inputMemAlign64 = p_logits_64 % vecAlignment_64 == 0 && p_gradients_64 % vecAlignment_64 == 0;

                if (wkts.GetDenominatorsSize() > 1 << (16 - sizeof(float))) {// 2^12 is the threshold
                    Tensor logits_logsoftmax = at::log_softmax(logits, 2, logits.scalar_type());
                    ctcloss_tools::ComputeCtcLoss</*DTYPE=*/c10::Half, /*CAST_DTYPE=*/float, /*VEC_DTYPE=*/VEC_DTYPE>(
                      /*logits=*/logits.data_ptr<c10::Half>(),
                      /*targets=*/targets.data_ptr<int>(),
                      /*logit_lengths=*/logit_lengths.data_ptr<int>(),
                      /*target_lengths=*/target_lengths.data_ptr<int>(),
                      /*int_workspace=*/int_workspace.data_ptr<int>(),
                      /*alpha_beta_workspace=*/alpha_beta_workspace.data_ptr<float>(),
                      /*float_workspace=*/float_workspace.data_ptr<float>(),
                      /*costs=*/costs.data_ptr<c10::Half>(),
                      /*costs_sum=*/costs_sum.data_ptr<c10::Half>(),
                      /*gradients=*/gradients.data_ptr<c10::Half>(),
                      /*CtcWorkspaceTools=*/&wkts,
                      /*inputMemAlign64=*/inputMemAlign64,
                      /*compute_sum=*/compute_sum,
                      /*compute_mean=*/compute_mean,
                      /*already_log_softmax=*/true);
                } else {
                    ctcloss_tools::ComputeCtcLoss</*DTYPE=*/c10::Half, /*CAST_DTYPE=*/float, /*VEC_DTYPE=*/VEC_DTYPE>(
                      /*logits=*/logits.data_ptr<c10::Half>(),
                      /*targets=*/targets.data_ptr<int>(),
                      /*logit_lengths=*/logit_lengths.data_ptr<int>(),
                      /*target_lengths=*/target_lengths.data_ptr<int>(),
                      /*int_workspace=*/int_workspace.data_ptr<int>(),
                      /*alpha_beta_workspace=*/alpha_beta_workspace.data_ptr<float>(),
                      /*float_workspace=*/float_workspace.data_ptr<float>(),
                      /*costs=*/costs.data_ptr<c10::Half>(),
                      /*costs_sum=*/costs_sum.data_ptr<c10::Half>(),
                      /*gradients=*/gradients.data_ptr<c10::Half>(),
                      /*CtcWorkspaceTools=*/&wkts,
                      /*inputMemAlign64=*/inputMemAlign64,
                      /*compute_sum=*/compute_sum,
                      /*compute_mean=*/compute_mean,
                      /*already_log_softmax=*/false);
                }
                break;
            }

            default: {
                break;
            }
        }

        if (logits_time_major) {// input logit size (T,N,C)
            gradients = gradients.permute({1, 0, 2}).contiguous();
        }

        if (compute_sum or compute_mean) { return std::make_tuple(costs_sum, gradients); }
        return std::make_tuple(costs, gradients);
    }

    class ctclossFunction : public torch::autograd::Function<ctclossFunction> {
      public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx, torch::autograd::Variable &logits, const at::Tensor &targets,
                                                      const at::Tensor &logit_lengths, const at::Tensor &target_lengths, int blank, double clamp, bool logits_time_major,
                                                      bool compute_sum, bool compute_mean, bool check_lengths) {
            auto result = compute_accelerate_ctc_loss(logits, targets, logit_lengths, target_lengths, blank, clamp, logits_time_major, compute_sum, compute_mean,
                                                      check_lengths);// result: costs, grads
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

            return {grads,
                    torch::autograd::Variable(),
                    torch::autograd::Variable(),
                    torch::autograd::Variable(),
                    torch::autograd::Variable(),
                    torch::autograd::Variable(),
                    torch::autograd::Variable(),
                    torch::autograd::Variable(),
                    torch::autograd::Variable(),
                    torch::autograd::Variable()};
        }
    };

    Tensor accelerate_ctc_loss(Tensor &logits, const Tensor &targets, const Tensor &logit_lengths, const Tensor &target_lengths, int blank, double clamp, bool logits_time_major,
                               bool compute_sum, bool compute_mean, bool check_lengths) {
        auto costs = ctclossFunction::apply(logits, targets, logit_lengths, target_lengths, blank, clamp, logits_time_major, compute_sum, compute_mean, check_lengths);
        return costs[0];
    }
  }
}// namespace at::native
