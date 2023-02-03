#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
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

#if defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#else

#include <cuda_fp16.h>
#include <cuda.h>

#endif

#include "loss_ctcloss.cuh"

namespace at {
  namespace native {

    namespace ctcloss_tools {

      // kernel: get max, from torchaudio
      template<int NUM_THREADS, typename DTYPE, typename CAST_DTYPE>
      __global__ void ReduceMax2D(
          int dim,
          const DTYPE *inputs, // [N, dim]
          CAST_DTYPE *outputs) {

        __shared__ CAST_DTYPE shared[NUM_THREADS];

        // each thread reduces one matrix row
        int offset = blockIdx.x * dim; // [n, 0]
        CAST_DTYPE val = CAST_DTYPE(inputs[offset]); // default = inputs(n, 0)
        for (int d = threadIdx.x; d < dim; d += NUM_THREADS) {
          CAST_DTYPE next = CAST_DTYPE(inputs[offset + d]);
          if (next > val) {
            val = next;
          }
        }

        shared[threadIdx.x] = val;
        __syncthreads();

        for (int stride = (NUM_THREADS >> 1); stride >= WARP_SIZE; stride >>= 1) {
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
            if (shf > val) {
              val = shf;
            }
          }
        }

        if (threadIdx.x == 0) {
          outputs[blockIdx.x] = val;
        }
      }

      template<int NUM_THREADS, typename DTYPE, typename CAST_DTYPE>
      __global__ void ReduceLogSumExpGivenMax2D(
          int dim,
          const DTYPE *inputs, // [N, dim]
          CAST_DTYPE *outputs) { // in: max -> out: logsum

        __shared__ CAST_DTYPE shared[NUM_THREADS];

        CAST_DTYPE max = outputs[blockIdx.x];
        CAST_DTYPE val = 0;

        int offset = blockIdx.x * dim;
        for (int d = threadIdx.x; d < dim; d += NUM_THREADS) {
          val += c10::cuda::compat::exp(CAST_DTYPE(inputs[offset + d]) - max);
        }

        shared[threadIdx.x] = val;
        __syncthreads();

        for (int stride = (NUM_THREADS >> 1); stride >= WARP_SIZE; stride >>= 1) {
          if (threadIdx.x < stride && threadIdx.x + stride < dim) {
            val = shared[threadIdx.x] + shared[threadIdx.x + stride];
            shared[threadIdx.x] = val;
          }
          __syncthreads();
        }

        CAST_DTYPE shf;
        for (int stride = (WARP_SIZE >> 1); stride > 0; stride >>= 1) {
          shf = __shfl_down(val, stride);
          if (threadIdx.x < stride && threadIdx.x + stride < dim) {
            val = val + shf;
          }
        }

        if (threadIdx.x == 0) {
          outputs[blockIdx.x] = -max - c10::cuda::compat::log(val);  // logits - max - log(sum(exp(logits-max)))
        }
      }

      template<int NUM_THREADS_FOR_LOG, typename DTYPE, typename CAST_DTYPE>
      void LogSumExp2D(
          cudaStream_t stream,
          int sizeBTU,
          int numTgt,
          const DTYPE *logits, //  input logits, size:[N, dim]
          CAST_DTYPE *outputs) {

        { // compute max among numTgt.
          dim3 block_dims(sizeBTU);
          dim3 thread_dims(NUM_THREADS_FOR_LOG);
          ReduceMax2D<NUM_THREADS_FOR_LOG, DTYPE, CAST_DTYPE>
          <<<block_dims, thread_dims, 0, stream>>>(
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

        { // compute log(sum(exp(d_i - max)))
          dim3 block_dims(sizeBTU);
          dim3 thread_dims(NUM_THREADS_FOR_LOG);
          ReduceLogSumExpGivenMax2D<NUM_THREADS_FOR_LOG, DTYPE, CAST_DTYPE>
          <<<block_dims, thread_dims, 0, stream>>>(
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

      template<typename CAST_DTYPE>
      __global__ void ComputeRepeatsAndLabelswithblank(
          int blankIdx,
          int maxTgtLen,
          int maxTgtWithBlankLen,
          const int *targets, // Size: {batchSize, maxTgtLen}
          const int *tgtLengths, // Size: {batchSize}
          int *pTargetsWithBlank, // Size: {batchSize, maxTgtWithBlankLen }
          int *pTargetRepeats) { // Size: {batchSize}

        int uid = threadIdx.x;
        int batch = blockIdx.x;

        const int tgtLen = tgtLengths[batch];
        const int *tgtVec = targets + batch * maxTgtLen;
        int *tgtWithBlankVec = pTargetsWithBlank + batch * maxTgtWithBlankLen;

        if (uid > tgtLen)
          return;

        for (int u = uid; u <= tgtLen; u += blockDim.x) {
          // set value, u <= tgtLen
          tgtWithBlankVec[2 * u] = blankIdx;
          if (u < tgtLen) { // u <= tgtLen-1
            tgtWithBlankVec[2 * u + 1] = tgtVec[u];

            if (u > 0 and tgtVec[u - 1] == tgtVec[u]) {   // repeat label
              pTargetRepeats[batch]++;
            }
          }
        }
      }

      template<typename DTYPE, typename CAST_DTYPE>
      __global__ void ComputeAlphasBetasCosts(
          const int batchSize,
          const int maxSrcLen,
          const int maxTgtLen,
          const int maxTgtWithBlankLen,
          const int numTargets,
          const int blankIdx,
          const DTYPE *logits, // Size: {batchSize, maxSrcLen, numTargets}
          const int *pTargetsWithBlank, // Size: {batchSize, maxTgtWithBlankLen}
          const int *srcLengths, // Size: {batchSize}
          const int *tgtLengths, // Size: {batchSize}
          const CAST_DTYPE *denominators, // Size: {batchSize, maxSrcLen}
          const int *pTargetRepeats, // Size: {batchSize}
          CAST_DTYPE *alphas, // Size: {batchSize, maxSrcLen, maxTgtWithBlankLen}
          CAST_DTYPE *betas, // Size: {batchSize, maxSrcLen, maxTgtWithBlankLen}
          DTYPE *costs,
          const bool compute_mean,
          const CAST_DTYPE neginf) {

        const int uid = threadIdx.x;  // target u
        const int batch = blockIdx.y;
        const int srcLen = srcLengths[batch];
        const int tgtLen = tgtLengths[batch]; // Note that start of the sentence is added as 1 here
        const int tgtWithBlankLen = 2 * tgtLen + 1;
        // get tgt
        const int *tgtWithBlankVec = pTargetsWithBlank + batch * maxTgtWithBlankLen;
        // get logProb
        const int batchOffset = batch * maxSrcLen; // NTC
        const DTYPE *logit = logits + batchOffset * numTargets;
        const CAST_DTYPE *denomVec = denominators + batchOffset;
        const int tgt_repeat = pTargetRepeats[batch];
        // alpha path
        CAST_DTYPE *batchAlpha = alphas + batchOffset * maxTgtWithBlankLen; // Size: {batchSize, maxSrcLen, maxTgtWithBlankLen}
        // beta path
        CAST_DTYPE *batchBeta = betas + batchOffset * maxTgtWithBlankLen; // Size: {batchSize, maxSrcLen, maxTgtWithBlankLen}

        // set Indexer
        Indexer2D idxer2d_src(numTargets);
        Indexer2D idxer2d_tgt(maxTgtWithBlankLen);

        if (tgtLen + tgt_repeat > srcLen) {
          return;
        }

        if (uid >= tgtWithBlankLen) { return; }

        if (blockIdx.x == 0) { // compute alpha

          // get log probs of label
          if (uid == 0) {
            batchAlpha[idxer2d_tgt(0, 0)] = CAST_DTYPE(logit[idxer2d_src(0, tgtWithBlankVec[0])]) + denomVec[0];
            batchAlpha[idxer2d_tgt(0, 1)] = CAST_DTYPE(logit[idxer2d_src(0, tgtWithBlankVec[1])]) + denomVec[0];
          }

          for (int t = 1; t < srcLen; ++t) {
            __syncthreads();

#pragma unroll
            for (int u = uid; u < tgtWithBlankLen; u += blockDim.x) {
              CAST_DTYPE alpha_t_i;
              if (u >= 1) {
                if (u >= 2 and tgtWithBlankVec[u] != blankIdx and tgtWithBlankVec[u] != tgtWithBlankVec[u - 2]) {
                  alpha_t_i = logSumExp3Input(batchAlpha[idxer2d_tgt(t - 1, u)], batchAlpha[idxer2d_tgt(t - 1, u - 1)], batchAlpha[idxer2d_tgt(t - 1, u - 2)], neginf);
                } else {
                  alpha_t_i = logSumExp2Input(batchAlpha[idxer2d_tgt(t - 1, u)], batchAlpha[idxer2d_tgt(t - 1, u - 1)], neginf);
                }
              } else {
                alpha_t_i = batchAlpha[idxer2d_tgt(t - 1, u)];
              }

              batchAlpha[idxer2d_tgt(t, u)] = alpha_t_i + CAST_DTYPE(logit[idxer2d_src(t, tgtWithBlankVec[u])]) + denomVec[t];
            }
          }
        } else if (blockIdx.x == 1) { // compute beta

          if (uid == 0) {
            batchBeta[idxer2d_tgt(srcLen - 1, tgtWithBlankLen - 1)] = CAST_DTYPE(logit[idxer2d_src(srcLen - 1, tgtWithBlankVec[tgtWithBlankLen - 1])]) + denomVec[srcLen - 1];
            batchBeta[idxer2d_tgt(srcLen - 1, tgtWithBlankLen - 2)] = CAST_DTYPE(logit[idxer2d_src(srcLen - 1, tgtWithBlankVec[tgtWithBlankLen - 2])]) + denomVec[srcLen - 1];
          }

          for (int t = srcLen - 2; t >= 0; --t) {
            __syncthreads();

#pragma unroll
            for (int u = uid; u < tgtWithBlankLen; u += blockDim.x) {
              CAST_DTYPE beta_t_i;
              if (u < (tgtWithBlankLen - 1)) {
                if (u < (tgtWithBlankLen - 2) and tgtWithBlankVec[u] != blankIdx and tgtWithBlankVec[u] != tgtWithBlankVec[u + 2]) {
                  beta_t_i = logSumExp3Input(batchBeta[idxer2d_tgt(t + 1, u)], batchBeta[idxer2d_tgt(t + 1, u + 1)], batchBeta[idxer2d_tgt(t + 1, u + 2)], neginf);
                } else {
                  beta_t_i = logSumExp2Input(batchBeta[idxer2d_tgt(t + 1, u)], batchBeta[idxer2d_tgt(t + 1, u + 1)], neginf);
                }
              } else {
                beta_t_i = batchBeta[idxer2d_tgt(t + 1, u)];
              }

              batchBeta[idxer2d_tgt(t, u)] = beta_t_i + CAST_DTYPE(logit[idxer2d_src(t, tgtWithBlankVec[u])]) + denomVec[t];
            }
          }

          if (uid == 0) {
            // loss = beta(0,0) + beta(0,1)
            costs[batch] = -DTYPE(logSumExp2Input(batchBeta[idxer2d_tgt(0, 0)], batchBeta[idxer2d_tgt(0, 1)], neginf));
          }
        }
      }

      template<typename DTYPE, typename CAST_DTYPE>
      __global__ void ComputeAlphaBetaSum(
          const int batchSize,
          const int maxSrcLen,
          const int maxTgtLen,
          const int maxTgtWithBlankLen,
          const int numTargets,
          const int *pTargetsWithBlank, // Size: {batchSize, maxTgtLen}
          const DTYPE *logits, // Size: {batchSize, maxSrcLen, numTargets}
          const CAST_DTYPE *denominators, // Size: {batchSize, maxSrcLen}
          const int *srcLengths, // Size: {batchSize}
          const int *tgtLengths, // Size: {batchSize}
          const DTYPE *costs, // Size: {batchSize}
          const CAST_DTYPE *alphas, // Size: {batchSize, maxSrcLen, maxTgtWithBlankLen}
          const CAST_DTYPE *betas, // Size: {batchSize, maxSrcLen, maxTgtWithBlankLen}
          CAST_DTYPE *pGradientAlphasBetas) { // Size: {batchSize, maxSrcLen, numTargets}

        const int u = threadIdx.x + blockIdx.z * blockDim.x; // part of tgtWithBlankLen
        const int t = blockIdx.x; // 0 ~ maxSrcLen
        const int batch = blockIdx.y; // 0 ~ batchSize

        if (batch >= batchSize) { return; }

        const int srcLen = srcLengths[batch];
        const int tgtLen = tgtLengths[batch];
        const int tgtWithBlankLen = 2 * tgtLen + 1;

        if (t >= srcLen) { return; }
        if (u >= tgtWithBlankLen) { return; }

        const int batchOffset = batch * maxSrcLen; // NTD
        const DTYPE cost = costs[batch];
        const CAST_DTYPE *batchAlpha = alphas + (batchOffset + t) * maxTgtWithBlankLen;
        const CAST_DTYPE *batchBeta = betas + (batchOffset + t) * maxTgtWithBlankLen;
        const int *tgtWithBlankVec = pTargetsWithBlank + batch * maxTgtWithBlankLen;
        const DTYPE *logit = logits + (batchOffset + t) * numTargets;
        const CAST_DTYPE denom_value = *(denominators + batchOffset + t);

        CAST_DTYPE *pAlphasBetasVec = pGradientAlphasBetas + (batchOffset + t) * numTargets; // already initialized to zeros
        int label = tgtWithBlankVec[u];

        gpuAtomicAddNoReturn(&pAlphasBetasVec[label], c10::cuda::compat::exp(batchAlpha[u] + batchBeta[u] + CAST_DTYPE(cost) - (CAST_DTYPE(logit[label]) + denom_value)));
      }

      template<typename DTYPE, typename CAST_DTYPE, typename VEC_DTYPE, int VF>
      __global__ void ComputeGradientsVec(
          const int batchSize,
          const int maxSrcLen,
          const int maxTgtLen,
          const int maxTgtWithBlankLen,
          const int numTargets,
          const int blankIdx,
          const CAST_DTYPE clamp,
          const DTYPE *logits, // Size: {batchSize, maxSrcLen, numTargets}
          const int *srcLengths, // Size: {batchSize}
          const int *tgtLengths, // Size: {batchSize}
          const CAST_DTYPE *denominators, // Size: {batchSize, maxSrcLen}
          const DTYPE *costs, // Size: {batchSize}
          CAST_DTYPE *pGradientAlphasBetas, // Size: {batchSize, maxSrcLen, numTargets}
          DTYPE *gradients, // Size: {batchSize, maxSrcLen, numTargets}
          const bool compute_mean) { // Size: {batchSize, maxSrcLen, numTargets}

        const int tid = threadIdx.x; // part of numTargets
        const int t = blockIdx.x; // 0 ~ maxSrcLen
        const int batch = blockIdx.y; // 0 ~ batchSize

        const int srcLen = srcLengths[batch];
        const int tgtLen = tgtLengths[batch];
        const int tgtWithBlankLen = 2 * tgtLen + 1;

        if (tid >= numTargets) { return; }

        const int batchOffset = batch * maxSrcLen; // NTC
        const DTYPE *logit = logits + (batchOffset + t) * numTargets;
        const CAST_DTYPE denom_value = *(denominators + batchOffset + t);
        const DTYPE cost = costs[batch];

        CAST_DTYPE *pAlphasBetasVec = pGradientAlphasBetas + (batchOffset + t) * numTargets;
        DTYPE *batchGrad = gradients + (batchOffset + t) * numTargets;

        // Variabels for vectorization
        DTYPE logitBuffer[VF], gradBuffer[VF];
        auto logitVec = reinterpret_cast<VEC_DTYPE const *>(logit); // using 64bit type
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
            CAST_DTYPE resGrad = 0; // exp(log_logits)
            if (pAlphasBetasVec[d] != INFINITY and pAlphasBetasVec[d] != 0) {
              resGrad = c10::cuda::compat::exp(CAST_DTYPE(logit[d]) + denom_value) - pAlphasBetasVec[d];
            }

            // compute mean and clamp
            if (compute_mean) {
              resGrad /= CAST_DTYPE(batchSize * maxTgtLen);
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
      }

      template<typename DTYPE, typename CAST_DTYPE>
      __global__ void ComputeGradients(
          const int batchSize,
          const int maxSrcLen,
          const int maxTgtLen,
          const int maxTgtWithBlankLen,
          const int numTargets,
          const int blankIdx,
          const CAST_DTYPE clamp,
          const DTYPE *logits, // Size: {batchSize, maxSrcLen, numTargets}
          const int *srcLengths, // Size: {batchSize}
          const int *tgtLengths, // Size: {batchSize}
          const CAST_DTYPE *denominators, // Size: {batchSize, maxSrcLen}
          const DTYPE *costs, // Size: {batchSize}
          CAST_DTYPE *pGradientAlphasBetas, // Size: {batchSize, maxSrcLen, numTargets}
          DTYPE *gradients, // Size: {batchSize, maxSrcLen, numTargets}
          const bool compute_mean) { // Size: {batchSize, maxSrcLen, numTargets}

        const int tid = threadIdx.x; // part of numTargets
        const int t = blockIdx.x; // 0 ~ maxSrcLen
        const int batch = blockIdx.y; // 0 ~ batchSize

        const int srcLen = srcLengths[batch];
        const int tgtLen = tgtLengths[batch];
        const int tgtWithBlankLen = 2 * tgtLen + 1;

        if (tid >= numTargets) { return; }

        const int batchOffset = batch * maxSrcLen; // NTD
        const DTYPE *logit = logits + (batchOffset + t) * numTargets;
        const CAST_DTYPE denom_value = *(denominators + batchOffset + t);
        const DTYPE cost = costs[batch];

        DTYPE *batchGrad = gradients + (batchOffset + t) * numTargets;
        CAST_DTYPE *pAlphasBetasVec = pGradientAlphasBetas + (batchOffset + t) * numTargets; // already initialized to zeros

        for (int d = tid; d < numTargets; d += blockDim.x) {
          // Do the update
          CAST_DTYPE resGrad = 0; // exp(logits) - exp(alpha+beta-cost-logits)
          if (pAlphasBetasVec[d] != INFINITY and pAlphasBetasVec[d] != 0) {
            resGrad = c10::cuda::compat::exp(CAST_DTYPE(logit[d]) + denom_value) - pAlphasBetasVec[d];
          }

          if (compute_mean) {
            resGrad /= CAST_DTYPE(batchSize * maxTgtLen);
          }
          if (clamp > 0) {
            resGrad = resGrad < clamp ? resGrad : clamp;
            resGrad = resGrad > (-clamp) ? resGrad : (-clamp);
          }

          batchGrad[d] = DTYPE(resGrad);
        }
      }

      template<int NUM_THREADS, typename DTYPE, typename CAST_DTYPE>
      __global__ void ComputeCostsFinal(
          int batchSize,
          int maxTgtLen,
          const DTYPE *costs,
          DTYPE *costs_sum,
          const bool compute_sum,
          const bool compute_mean) {

        __shared__ CAST_DTYPE shared[NUM_THREADS];

        // each thread reduces one matrix row
        CAST_DTYPE val = CAST_DTYPE(0); // default = inputs(n, 0)
        for (int d = threadIdx.x; d < batchSize; d += NUM_THREADS) {
          CAST_DTYPE next = CAST_DTYPE(costs[d]);
          val += next;
        }

        shared[threadIdx.x] = val;
        __syncthreads();

        for (int stride = (NUM_THREADS >> 1); stride >= WARP_SIZE; stride >>= 1) {
          if (threadIdx.x < stride && threadIdx.x + stride < batchSize) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
            val = shared[threadIdx.x];
          }
          __syncthreads();
        }

        CAST_DTYPE shf;
        for (int stride = (WARP_SIZE >> 1); stride > 0; stride >>= 1) {
          shf = __shfl_down(val, stride);
          if (threadIdx.x < stride && threadIdx.x + stride < batchSize) {
            val += shf;
          }
        }

        if (threadIdx.x == 0) {
          if (compute_mean) {
            costs_sum[0] = DTYPE(val / (batchSize * maxTgtLen));
          } else {
            costs_sum[0] = DTYPE(val);
          }
        }
      }

//=============================================================================================//
/* check length of srcLengths and tgtLengths */
      void LengthsCheck(CtcWorkspaceTools *wkts,
                        const int *srcLengths, const int *tgtLengths) {
        int maxSrcLengths = 0, maxTgtLengths = 0;
        for (int b = 0; b < wkts->GetBatchSize(); ++b) {
          if (maxSrcLengths < srcLengths[b]) maxSrcLengths = srcLengths[b];
          if (maxTgtLengths < tgtLengths[b]) maxTgtLengths = tgtLengths[b];
        }
        TORCH_CHECK(wkts->GetMaxSrcLen() == maxSrcLengths, "input length mismatch");
        TORCH_CHECK(wkts->GetMaxTgtLen() == maxTgtLengths, "target length mismatch");
      }

      template<typename DTYPE, typename CAST_DTYPE, typename VEC_DTYPE>
      void ComputeCtcLoss(const DTYPE *logits, const int *targets, const int *srcLengths, const int *tgtLengths,
                          int *intWorkspace, CAST_DTYPE *alphaBetaWorkspace, CAST_DTYPE *floatWorkspace,
                          DTYPE *costs, DTYPE *costs_sum, DTYPE *gradients,
                          CtcWorkspaceTools *wkts, const bool &inputMemAlign,
                          const bool &compute_sum, const bool &compute_mean, const bool &already_log_softmax) {

        const CAST_DTYPE neginf = -CAST_DTYPE(INFINITY);

        const cudaStream_t &stream = wkts->GetCudaStream();
        const int &batchSize = wkts->GetBatchSize();
        const int &maxSrcLen = wkts->GetMaxSrcLen();
        const int &maxTgtLen = wkts->GetMaxTgtLen();
        const int &maxTgtWithBlankLen = wkts->GetMaxTgtWithBlankLen();
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
              /*logits=*/logits, // size is (B,T,D)
              /*denominators=*/pDenominators
          );
        }

        int *pTargetRepeats = intWorkspace + wkts->GetPointerToTargetRepeatsOffset();
        int *pTargetsWithBlank = intWorkspace + wkts->GetLabelsWithBlankOffset();
        { // compute labels_with_blank and target_repeat
          const int tgtPerThread = (maxTgtLen + WORK_NUM_THREADS - 1) / WORK_NUM_THREADS;
          const dim3 thread_dims(std::min(MaxThreadPerElement, tgtPerThread)); // each thread deal a tgt vector
          const dim3 block_dims(batchSize);

          ComputeRepeatsAndLabelswithblank<CAST_DTYPE>
          <<<block_dims, thread_dims, 0, stream>>>(
              /*blankIdx=*/blank,
              /*maxTgtLen=*/maxTgtLen,
              /*maxTgtWithBlankLen=*/maxTgtWithBlankLen,
              /*targets=*/targets,
              /*tgtLengths=*/tgtLengths,
              /*pTargetsWithBlank=*/pTargetsWithBlank,
              /*pTargetRepeats=*/pTargetRepeats);

          if (cudaGetLastError() != cudaSuccess) {
            std::string msg("ComputeRepeatsAndLabelswithblank failed: ");
            throw std::runtime_error(msg);
          }
        }

        CAST_DTYPE *pAlphas = alphaBetaWorkspace + wkts->GetPointerToAlphasOffset();
        CAST_DTYPE *pBetas = alphaBetaWorkspace + wkts->GetPointerToBetasOffset();
        { // compute alphas, betas and costs.
          const int workNumThread = 1;
          const int tgtPerThread = (maxTgtWithBlankLen + workNumThread - 1) / workNumThread;
          const dim3 thread_dims(std::min(MAX_BLOCK_SIZE, tgtPerThread)); // each thread deal a tgt vector
          const dim3 block_dims(2, batchSize);

          ComputeAlphasBetasCosts<DTYPE, CAST_DTYPE>
          <<<block_dims, thread_dims, 0, stream>>>(
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

        CAST_DTYPE *pGradientAlphasBetas = floatWorkspace + wkts->GetPointerToGradientAlphasBetasOffset();
        {
          const int tgtPerThread = (maxTgtWithBlankLen + WORK_NUM_THREADS - 1) / WORK_NUM_THREADS;
          const int thread_size = std::min(MaxThreadPerElement, tgtPerThread);
          const int block_sizeZ = (maxTgtWithBlankLen - 1) / thread_size + 1;
          const dim3 thread_dims(thread_size); // each thread deal a tgt vector
          const dim3 block_dims(maxSrcLen, batchSize, block_sizeZ); // blocks deal vectors
          ComputeAlphaBetaSum<DTYPE, CAST_DTYPE>
          <<<block_dims, thread_dims, 0, stream>>>(
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
              /*pGradientAlphasBetas=*/pGradientAlphasBetas);

          if (cudaGetLastError() != cudaSuccess) {
            std::string msg("ComputeAlphaBetaSum failed: ");
            throw std::runtime_error(msg);
          }
        }

        { // compute gradients.
          const int tgtPerThread = (numTgt + WORK_NUM_THREADS - 1) / WORK_NUM_THREADS;
          const dim3 thread_dims(std::min(MaxThreadPerElement, tgtPerThread)); // each thread deal a tgt vector
          const dim3 block_dims(maxSrcLen, batchSize); // blocks deal vectors

          const int vecFactor = sizeof(VEC_DTYPE) / sizeof(DTYPE);
          bool typeAlign = numTgt % vecFactor == 0;

          if (typeAlign and inputMemAlign) {
            // Would like each thread to work on 4 hidden units
            ComputeGradientsVec<DTYPE, CAST_DTYPE, VEC_DTYPE, vecFactor>
            <<<block_dims, thread_dims, 0, stream>>>(
                /*batchSize=*/batchSize,
                /*maxSrcLen=*/maxSrcLen,
                /*maxTgtLen=*/maxTgtLen,
                /*maxTgtWithBlankLen=*/maxTgtWithBlankLen,
                /*num_targets=*/numTgt,
                /*blank=*/blank,
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
            ComputeGradients<DTYPE, CAST_DTYPE>
            <<<block_dims, thread_dims, 0, stream>>>(
                /*batchSize=*/batchSize,
                /*maxSrcLen=*/maxSrcLen,
                /*maxTgtLen=*/maxTgtLen,
                /*maxTgtWithBlankLen=*/maxTgtWithBlankLen,
                /*num_targets=*/numTgt,
                /*blank=*/blank,
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

        if (compute_sum or compute_mean) { // compute final cost.
          const dim3 block_dims(1);
          const dim3 thread_dims(REDUCE_THREADS);
          ComputeCostsFinal<REDUCE_THREADS, DTYPE, CAST_DTYPE>
          <<<block_dims, thread_dims, 0, stream>>>(
              /*batchSize=*/batchSize,
              /*maxTgtLen=*/maxTgtLen,
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

// Entry point into CTC Loss
    std::tuple <Tensor, Tensor> accelerate_ctc_loss(
        Tensor &logits,
        const Tensor &targets,
        const Tensor &logit_lengths,
        const Tensor &target_lengths,
        int blank,
        double clamp,
        bool logits_time_major,
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
      TORCH_CHECK(logits.dim() == 3, "logits must be 3-D (time, batch, class)");  // Tensor of size (T, N, C) or (N,T,C)
      TORCH_CHECK(targets.dim() == 2, "targets must be 2-D (batch, max target length)"); // Tensor of size (T, S)
      // It represent the target sequences. Each element in the target sequence is a class index. And the target index cannot be blank (default=0).
      // In the (N, S) form, targets are padded to the length of the longest sequence, and stacked.
      TORCH_CHECK(logit_lengths.dim() == 1, "logit_lengths must be 1-D");
      TORCH_CHECK(target_lengths.dim() == 1, "target_lengths must be 1-D");

      if (logits_time_major) { // input logit size (T,N,C)
        logits = logits.permute({1, 0, 2}).contiguous(); // (T,N,C) -> (N,T,C)
      }

      TORCH_CHECK(logit_lengths.size(0) == logits.size(0), "batch dimension mismatch between logits and logit_lengths");
      TORCH_CHECK(target_lengths.size(0) == logits.size(0), "batch dimension mismatch between logits and target_lengths");
      TORCH_CHECK(targets.size(0) == logits.size(0), "batch dimension mismatch between logits and targets");
      TORCH_CHECK(blank >= 0 && blank < logits.size(-1), "blank must be within [0, logits.shape[-1])");

      // Set options
      ctcloss_tools::CtcWorkspaceTools wkts(
          /*logits=*/logits,
          /*targets=*/targets,
          /*clamp=*/clamp,
          /*blank=*/blank);

      wkts.SetCudaStream(at::cuda::getCurrentCUDAStream());

      if (check_lengths) {
        // use cpu is faster than cuda
        ctcloss_tools::LengthsCheck(&wkts, logit_lengths.data_ptr<int>(), target_lengths.data_ptr<int>());
      }

      // create int workspace
      Tensor int_workspace = at::zeros(wkts.GetIntWorkspaceSize(),
                                       at::TensorOptions().device(logits.device()).dtype(ScalarType::Int));
      // create float workspace
      Tensor alpha_beta_workspace = at::empty(wkts.GetAlphaBetaWorkspaceSize(),
                                              at::TensorOptions().device(logits.device()).dtype(ScalarType::Float), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      alpha_beta_workspace.fill_(float(-INFINITY));
      alpha_beta_workspace = alpha_beta_workspace.contiguous();

      Tensor float_workspace = at::zeros(wkts.GetFloatWorkspaceSize(),
                                         at::TensorOptions().device(logits.device()).dtype(ScalarType::Float));

      Tensor gradients = at::zeros_like(logits, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      // set output
      Tensor costs = at::empty(wkts.GetBatchSize(), logits.options());
      Tensor costs_sum = at::zeros({1}, logits.options());

      // vector compute
      using VEC_DTYPE = uint64_t;
      constexpr int vecAlignment = std::alignment_of<VEC_DTYPE>::value;

      switch (logits.scalar_type()) {
        case ScalarType::Float: {

          bool inputMemAlign = reinterpret_cast<VEC_DTYPE>(logits.data_ptr<float>()) % vecAlignment == 0
                               and reinterpret_cast<VEC_DTYPE>(gradients.data_ptr<float>()) % vecAlignment == 0;

          if (wkts.GetDenominatorsSize() > 1 << (16 - sizeof(float))) { // 2^12 is the threshold
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
                /*inputMemAlign=*/inputMemAlign,
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

          if (wkts.GetDenominatorsSize() > 1 << (16 - sizeof(float))) { // 2^12 is the threshold
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
                /*inputMemAlign=*/inputMemAlign,
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
                /*inputMemAlign=*/inputMemAlign,
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

      if (logits_time_major) { // input logit size (T,N,C)
        gradients = gradients.permute({1, 0, 2}).contiguous();
      }

      if (compute_sum or compute_mean) {
        return std::make_tuple(costs_sum, gradients);
      }
      return std::make_tuple(costs, gradients);
    }
  }
} // namespace at::native
