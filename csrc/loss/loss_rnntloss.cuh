// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef LIGHTOP_LOSS_RNNTLOSS_CUH
#define LIGHTOP_LOSS_RNNTLOSS_CUH

#include <THC/THCDeviceUtils.cuh>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#include <ATen/native/cuda/LaunchUtils.h>
#include <c10/macros/Macros.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace at {
  namespace native {
    namespace rnntloss_tools {
      constexpr int REDUCE_THREADS = 256;
      constexpr int MAX_BLOCK_SIZE = 1024;
      constexpr int MAX_GRID_SIZE = 65536;
      constexpr int WARP_SIZE = 64;
      constexpr int WORK_NUM_THREADS = 4; // Number of threads executing the same work
      const int MaxThreadPerElement = int(MAX_BLOCK_SIZE / WORK_NUM_THREADS);  // 256
//      const auto deviceProperties = at::cuda::getCurrentDeviceProperties();
//      constexpr int MAX_BLOCK_SIZE = deviceProperties->maxThreadsPerBlock; // 1024
//      constexpr int MAX_GRID_SIZE = deviceProperties->sharedMemPerBlock; // 65536
//      constexpr int WARP_SIZE = deviceProperties->warpSize; // 64

      ////////////////////////////////////////////////////////////////////////////////
#define HOST_AND_DEVICE __host__ __device__
#define FORCE_INLINE __forceinline__

      struct Indexer2D {
        const int &size2_;

        FORCE_INLINE HOST_AND_DEVICE Indexer2D(const int &size2) : size2_(size2) {}

        FORCE_INLINE HOST_AND_DEVICE int operator()(int index1, int index2) {
          return index1 * size2_ + index2;
        }
      };

      struct Indexer3D {
        const int &size2_;
        const int &size3_;

        FORCE_INLINE HOST_AND_DEVICE Indexer3D(const int &size2, const int &size3)
            : size2_(size2), size3_(size3) {}

        FORCE_INLINE HOST_AND_DEVICE int operator()(
            int index1,
            int index2,
            int index3) {
          return (index1 * size2_ + index2) * size3_ + index3;
        }
      };

      // math tool
      template<typename scalar_t>
      __device__ __forceinline__ scalar_t logSumExp(scalar_t a, scalar_t b) {
        // standard log-sum-exp trick is used here to provide better numerical stability
        return (a >= b) ? a + c10::cuda::compat::log1p(c10::cuda::compat::exp(b - a)) :
               b + c10::cuda::compat::log1p(c10::cuda::compat::exp(a - b));
      }

      /*
       * Related tool functions
       * */
      class RnntWorkspaceTools {
      public:
          RnntWorkspaceTools() : batchSize_(0), maxSrcLen_(0), maxTgtLen_(0), numTargets_(0),
                                 blank_(-1), clamp_(-1) {}

          RnntWorkspaceTools(Tensor &logits, double &clamp, int &blank) :
              batchSize_(logits.size(0)), maxSrcLen_(logits.size(1)), maxTgtLen_(logits.size(2)),
              numTargets_(logits.size(3)), clamp_(clamp), blank_(blank) {
            // Size and workspace
            sizeBTU_ = batchSize_ * maxSrcLen_ * maxTgtLen_;
          }

          //// Using workspace from torchaudio
          // get Size
          int GetDenominatorsSize() const { return sizeBTU_; }

          int GetAlphasOrBetasSize() const { return sizeBTU_; }

          int GetFloatWorkspaceSize() const {
            return GetDenominatorsSize() + GetAlphasOrBetasSize() * 2;
          }

          // get Offset
          int GetPointerToDenominatorsOffset() const { return 0; }

          int GetPointerToAlphasOffset() const { return GetPointerToDenominatorsOffset() + GetDenominatorsSize(); }

          int GetPointerToBetasOffset() const { return GetPointerToAlphasOffset() + GetAlphasOrBetasSize(); }

          int GetBatchSize() { return batchSize_; }

          int GetMaxSrcLen() { return maxSrcLen_; }

          int GetMaxTgtLen() { return maxTgtLen_; }

          int GetNumTargets() { return numTargets_; }

          int GetBlankIdx() { return blank_; }

          double GetClamp() { return clamp_; }

          void SetCudaStream(cudaStream_t stream) { stream_ = stream; }

          cudaStream_t GetCudaStream() { return stream_; }

      private:
          // the stream to launch kernels in when using GPU.
          cudaStream_t stream_;
          // the index for "blank".
          int blank_;
          // gradient clamp value.
          double clamp_; // negative for disabling clamping by default.

          // batch size = B.
          int batchSize_;
          // the maximum length of src encodings = max_T.
          int maxSrcLen_;
          // the maximum length of tgt encodings = max_U.
          int maxTgtLen_;
          // num_targets = D.
          int numTargets_;

          // common size
          int sizeBTU_;
      };

    } // namespace rnntloss_tools
  }
} // namespace at::native

#endif //LIGHTOP_LOSS_RNNTLOSS_CUH
