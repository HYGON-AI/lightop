#include "sampling/sampling.h"
#include "sampling/sampling_impl.hpp"

namespace lightop {
namespace sampling {

constexpr BlockScanAlgorithm SCAN_ALGO = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;

static constexpr int kBlockThreads = 256;

template <typename T, typename IdType>
void TopKSamplingFromProb(T* probs, IdType* output, IdType* indices, IdType* top_k_arr,
                          uint32_t batch_size, uint32_t top_k_val, uint32_t d,
                          bool deterministic, uint64_t philox_seed, uint64_t philox_offset,
                          hipStream_t stream = 0) {
  constexpr int VEC_SIZE = 16 / sizeof(T);

  const uint32_t smem_size = sizeof(SamplingTempStorage<kBlockThreads, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(kBlockThreads);

  DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
    auto kernel =
      TopKSamplingFromProbKernel<kBlockThreads, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE, DETERMINISTIC, T, IdType>;
    kernel<<<nblks, nthrs, smem_size, stream>>>(
      probs, output, indices, top_k_arr, top_k_val, d, philox_seed, philox_offset);
  });
}

void top_k_sampling_from_probs(const torch::Tensor& probs, torch::Tensor& output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset) {
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  using IdType = int;
  hipStream_t stream = at::cuda::getCurrentHIPStream();

  sampling::TopKSamplingFromProb<float, IdType>(
    static_cast<float*>(probs.data_ptr()), static_cast<IdType*>(output.data_ptr()),
    maybe_indices.has_value()
      ? static_cast<IdType*>(maybe_indices.value().data_ptr())
      : nullptr,
    has_top_k_arr ? static_cast<IdType*>(maybe_top_k_arr.value().data_ptr()) : nullptr,
    batch_size, top_k_val, vocab_size, deterministic, philox_seed, philox_offset, stream);
}

template <typename T, typename IdType>
void TopPSamplingFromProb(T* probs, IdType* output, IdType* indices, T* top_p_arr,
                          uint32_t batch_size, T top_p_val, uint32_t d, bool deterministic,
                          uint64_t philox_seed, uint64_t philox_offset,
                          hipStream_t stream = 0) {
  const uint32_t VEC_SIZE = 16 / sizeof(T);

  const uint32_t smem_size = sizeof(SamplingTempStorage<kBlockThreads, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(kBlockThreads);

  DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
    auto kernel =
      TopPSamplingFromProbKernel<kBlockThreads, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,DETERMINISTIC, T, IdType>;
    kernel<<<nblks, nthrs, smem_size, stream>>>(
      probs, output, indices, top_p_arr, top_p_val, d, philox_seed, philox_offset);
  });
}

void top_p_sampling_from_probs(const torch::Tensor& probs, torch::Tensor& output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_p_arr, float top_p_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset) {
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  using IdType = int;
  hipStream_t stream = at::cuda::getCurrentHIPStream();

  sampling::TopPSamplingFromProb<float, IdType>(
        static_cast<float*>(probs.data_ptr()), static_cast<IdType*>(output.data_ptr()),
        maybe_indices.has_value() ? static_cast<IdType*>(maybe_indices.value().data_ptr())
                                  : nullptr,
        has_top_p_arr ? static_cast<float*>(maybe_top_p_arr.value().data_ptr()) : nullptr,
        batch_size, top_p_val, vocab_size, deterministic, philox_seed, philox_offset, stream);
}

template <typename T, typename IdType>
void TopKTopPSamplingFromProb(T* probs, IdType* top_k_arr, T* top_p_arr, IdType* output,
                              IdType* indices, uint32_t batch_size, IdType top_k_val,
                              T top_p_val, uint32_t d, bool deterministic,
                              uint64_t philox_seed, uint64_t philox_offset,
                              hipStream_t stream = 0) {
  constexpr uint32_t VEC_SIZE = 16 / sizeof(T);

  const uint32_t smem_size = sizeof(SamplingTempStorage<kBlockThreads, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(kBlockThreads);

  DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
    auto kernel = TopKTopPSamplingFromProbKernel<
      kBlockThreads, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE, DETERMINISTIC, T, IdType>;
    kernel<<<nblks, nthrs, smem_size, stream>>>(
      probs, top_k_arr, top_p_arr, output, indices, top_k_val, top_p_val, d, philox_seed, philox_offset);
  });
}

void top_k_top_p_sampling_from_probs(const torch::Tensor& probs, torch::Tensor& output,
                                     std::optional<torch::Tensor> maybe_indices,
                                     std::optional<torch::Tensor> maybe_top_k_arr, int64_t top_k_val,
                                     std::optional<torch::Tensor> maybe_top_p_arr, float top_p_val,
                                     bool deterministic, uint64_t philox_seed,
                                     uint64_t philox_offset) {
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  using IdType = int;
  hipStream_t stream = at::cuda::getCurrentHIPStream();

  sampling::TopKTopPSamplingFromProb<float, IdType>(
        static_cast<float*>(probs.data_ptr()),
        has_top_k_arr ? static_cast<IdType*>(maybe_top_k_arr.value().data_ptr()) : nullptr,
        has_top_p_arr ? static_cast<float*>(maybe_top_p_arr.value().data_ptr()) : nullptr,
        static_cast<IdType*>(output.data_ptr()),
        maybe_indices.has_value() ? static_cast<IdType*>(maybe_indices.value().data_ptr())
                                  : nullptr,
        batch_size, top_k_val, top_p_val, vocab_size, deterministic, philox_seed, philox_offset,
        stream);
}

}
}