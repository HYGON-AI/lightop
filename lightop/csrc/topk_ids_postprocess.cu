#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>

namespace at {
namespace native {
namespace {

constexpr int kInt32Dtype = 0;
constexpr int kInt64Dtype = 1;
constexpr int kThreads = 256;

__device__ __forceinline__ int64_t load_index_value(
    const void* ptr,
    int dtype,
    int64_t offset) {
  if (dtype == kInt32Dtype) {
    return static_cast<int64_t>(reinterpret_cast<const int32_t*>(ptr)[offset]);
  }
  return reinterpret_cast<const int64_t*>(ptr)[offset];
}

template <typename topk_t>
__global__ void topk_ids_postprocess_kernel(
    topk_t* __restrict__ topk_ids,
    const void* __restrict__ logical_to_physical_map,
    const void* __restrict__ num_token_non_padded,
    int64_t total,
    int64_t topk,
    int64_t map_size,
    int map_dtype,
    int num_token_dtype) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  if (num_token_non_padded != nullptr) {
    int64_t num_valid_tokens =
        load_index_value(num_token_non_padded, num_token_dtype, 0);
    int64_t token_idx = idx / topk;
    if (token_idx >= num_valid_tokens) {
      topk_ids[idx] = static_cast<topk_t>(-1);
      return;
    }
  }

  if (logical_to_physical_map == nullptr) {
    return;
  }

  int64_t expert_id = static_cast<int64_t>(topk_ids[idx]);
  int64_t physical_id = -1;
  if (expert_id >= 0 && expert_id < map_size) {
    physical_id = load_index_value(
        logical_to_physical_map, map_dtype, expert_id);
  }
  topk_ids[idx] = static_cast<topk_t>(physical_id);
}

int index_dtype_tag(const torch::Tensor& tensor, const char* name) {
  if (tensor.scalar_type() == at::ScalarType::Int) {
    return kInt32Dtype;
  }
  if (tensor.scalar_type() == at::ScalarType::Long) {
    return kInt64Dtype;
  }
  TORCH_CHECK(false, name, " expects int32 or int64 tensor");
  return kInt32Dtype;
}

}  // namespace

void topk_ids_postprocess(
    torch::Tensor topk_ids,
    torch::optional<torch::Tensor> logical_to_physical_map_opt,
    torch::optional<torch::Tensor> num_token_non_padded_opt) {
  if (topk_ids.numel() == 0) {
    return;
  }

  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids_postprocess expects CUDA tensors");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids_postprocess expects 2D topk_ids");
  TORCH_CHECK(
      topk_ids.is_contiguous(),
      "topk_ids_postprocess expects contiguous topk_ids");
  TORCH_CHECK(
      topk_ids.scalar_type() == at::ScalarType::Int ||
          topk_ids.scalar_type() == at::ScalarType::Long,
      "topk_ids_postprocess expects int32 or int64 topk_ids");

  const void* logical_to_physical_map = nullptr;
  int64_t map_size = 0;
  int map_dtype = kInt32Dtype;
  if (logical_to_physical_map_opt.has_value() &&
      logical_to_physical_map_opt->defined() &&
      logical_to_physical_map_opt->numel() > 0) {
    const auto& map = logical_to_physical_map_opt.value();
    TORCH_CHECK(map.is_cuda(), "topk_ids_postprocess expects CUDA map");
    TORCH_CHECK(
        map.device() == topk_ids.device(),
        "topk_ids_postprocess expects map on the same device as topk_ids");
    TORCH_CHECK(map.dim() == 1, "topk_ids_postprocess expects 1D map");
    TORCH_CHECK(map.is_contiguous(), "topk_ids_postprocess expects contiguous map");
    map_dtype = index_dtype_tag(map, "topk_ids_postprocess map");
    logical_to_physical_map = map.data_ptr();
    map_size = map.numel();
  }

  const void* num_token_non_padded = nullptr;
  int num_token_dtype = kInt32Dtype;
  if (num_token_non_padded_opt.has_value() &&
      num_token_non_padded_opt->defined() &&
      num_token_non_padded_opt->numel() > 0) {
    const auto& num_token = num_token_non_padded_opt.value();
    TORCH_CHECK(
        num_token.is_cuda(),
        "topk_ids_postprocess expects CUDA num_token_non_padded");
    TORCH_CHECK(
        num_token.device() == topk_ids.device(),
        "topk_ids_postprocess expects num_token_non_padded on the same device as topk_ids");
    TORCH_CHECK(
        num_token.is_contiguous(),
        "topk_ids_postprocess expects contiguous num_token_non_padded");
    num_token_dtype =
        index_dtype_tag(num_token, "topk_ids_postprocess num_token_non_padded");
    num_token_non_padded = num_token.data_ptr();
  }

  if (logical_to_physical_map == nullptr && num_token_non_padded == nullptr) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_ids));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t total = topk_ids.numel();
  int64_t topk = topk_ids.size(1);
  dim3 block(kThreads);
  dim3 grid((total + kThreads - 1) / kThreads);

  if (topk_ids.scalar_type() == at::ScalarType::Int) {
    topk_ids_postprocess_kernel<int32_t><<<grid, block, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(),
        logical_to_physical_map,
        num_token_non_padded,
        total,
        topk,
        map_size,
        map_dtype,
        num_token_dtype);
  } else {
    topk_ids_postprocess_kernel<int64_t><<<grid, block, 0, stream>>>(
        topk_ids.data_ptr<int64_t>(),
        logical_to_physical_map,
        num_token_non_padded,
        total,
        topk,
        map_size,
        map_dtype,
        num_token_dtype);
  }
}

}  // namespace native
}  // namespace at
