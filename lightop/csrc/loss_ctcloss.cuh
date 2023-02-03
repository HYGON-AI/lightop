#ifndef LIGHTOP_LOSS_CTCLOSS_CUH
#define LIGHTOP_LOSS_CTCLOSS_CUH

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


namespace at {
  namespace native {
    namespace ctcloss_tools {
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
      __device__ __forceinline__ scalar_t logSumExp2Input(scalar_t a, scalar_t b, scalar_t neginf) {
        scalar_t lmax = a >= b ? a : b;
        lmax = lmax == neginf ? 0 : lmax;
        return c10::cuda::compat::log(c10::cuda::compat::exp(a - lmax) + c10::cuda::compat::exp(b - lmax)) + lmax;
      }

      template<typename scalar_t>
      __device__ __forceinline__ scalar_t logSumExp3Input(scalar_t a, scalar_t b, scalar_t c, scalar_t neginf) {
        scalar_t lmax = a >= b ? a : b;
        lmax = lmax >= c ? lmax : c;
        lmax = lmax == neginf ? 0 : lmax;
        return c10::cuda::compat::log(c10::cuda::compat::exp(a - lmax) + c10::cuda::compat::exp(b - lmax) + c10::cuda::compat::exp(c - lmax)) + lmax;
      }

      /*
       * Related tool functions
       * */
      class CtcWorkspaceTools {
      public:
          CtcWorkspaceTools() : batchSize_(0), maxSrcLen_(0), maxTgtLen_(0), numTargets_(0),
                                blank_(-1), clamp_(-1) {}

          CtcWorkspaceTools(Tensor &logits, const Tensor &targets, double &clamp, int &blank) :
              batchSize_(logits.size(0)), maxSrcLen_(logits.size(1)), maxTgtLen_(targets.size(1)),
              numTargets_(logits.size(2)), clamp_(clamp), blank_(blank) {
            // Size and workspace
            sizeBT_ = batchSize_ * maxSrcLen_;
            maxTgtWithBlankLen_ = 2 * maxTgtLen_ + 1;
            sizeBTC_ = batchSize_ * maxSrcLen_ * numTargets_;
            sizeBTS_ = batchSize_ * maxSrcLen_ * maxTgtWithBlankLen_;
            sizeBS_ = batchSize_ * maxTgtWithBlankLen_;
          }

          //// Using workspace

          // DtypeWorkspace holds a "view" of  workspace for:
          //     1. softmax denominators (in log form), size = B * max_T
          //     2. alphas, size = B * max_T * maxTgtWithBlankLen
          //     3. betas, size = B * max_T * maxTgtWithBlankLen
          //     4. alphas and betas product, size = B * max_T * max_U

          // IntWorkspace holds a "view" of workspace for:
          //     1. labels_with_blank, size = B * max_T * numTargets
          //     2. labels repeats, size = B

          // get Size
          int GetDenominatorsSize() const { return sizeBT_; }

          // steps for computing alpha and beta
          int GetLabelsWithBlankSize() const { return sizeBS_; } // using float instead of int type

          // record repeat number for target
          int GetTargetRepeatsSize() const { return batchSize_; }

          int GetAlphasOrBetasSize() const { return sizeBTS_; }

          int GetGradientAlphasBetasSize() const { return sizeBTC_; }

          int GetAlphaBetaWorkspaceSize() const {
            return GetAlphasOrBetasSize() * 2;
          }

          int GetFloatWorkspaceSize() const {
            return GetDenominatorsSize() + GetGradientAlphasBetasSize();
          }

          int GetIntWorkspaceSize() const {
            return GetLabelsWithBlankSize() + GetTargetRepeatsSize();
          }

          // get float offset
          int GetPointerToAlphasOffset() const { return 0; }

          int GetPointerToBetasOffset() const { return GetPointerToAlphasOffset() + GetAlphasOrBetasSize(); }

          int GetPointerToDenominatorsOffset() const { return 0; }

          int GetPointerToGradientAlphasBetasOffset() const { return GetPointerToDenominatorsOffset() + GetDenominatorsSize(); }

          // get int offset
          int GetPointerToTargetRepeatsOffset() const { return 0; }

          int GetLabelsWithBlankOffset() const { return GetPointerToTargetRepeatsOffset() + GetTargetRepeatsSize(); }

          // get values
          int GetBatchSize() { return batchSize_; }

          int GetMaxSrcLen() { return maxSrcLen_; }

          int GetMaxTgtWithBlankLen() { return maxTgtWithBlankLen_; }

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
          // the maximum length of tgt encodings with blank = 2 * max_U + 1.
          int maxTgtWithBlankLen_;
          // num_targets = D.
          int numTargets_;

          // common size
          int sizeBT_, sizeBTC_, sizeBTS_, sizeBS_;
      };

    } // namespace ctcloss_tools

    extern Tensor fuse_log_softmax_forward(const at::Tensor &self, int64_t dim, bool half_to_float);

  }
} // namespace at::native

#endif //LIGHTOP_LOSS_CTCLOSS_CUH
