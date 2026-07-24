#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>

#include <cstdint>

namespace at {
namespace native {
namespace {

__device__ __forceinline__ int64_t ceil_div_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

__global__ void lightop_alloc_extend_kernel(
    const int64_t* __restrict__ pre_lens_ptr,
    const int64_t* __restrict__ seq_lens_ptr,
    const int64_t* __restrict__ last_loc_ptr,
    const int64_t* __restrict__ free_page_ptr,
    int64_t* __restrict__ out_indices,
    int64_t bs,
    int64_t page_size) {
  const int64_t pid = blockIdx.x;
  if (pid >= bs) {
    return;
  }

  __shared__ int64_t seq_len;
  __shared__ int64_t pre_len;
  __shared__ int64_t extend_len;
  __shared__ int64_t output_start_loc;
  __shared__ int64_t new_page_start_loc;
  __shared__ int64_t ceil_pre_pages;
  __shared__ int64_t last_loc;

  if (threadIdx.x == 0) {
    seq_len = seq_lens_ptr[pid];
    pre_len = pre_lens_ptr[pid];
    extend_len = seq_len - pre_len;

    int64_t sum_extend_lens = 0;
    int64_t sum_num_new_pages = 0;
    for (int64_t i = 0; i <= pid; ++i) {
      const int64_t other_seq_len = seq_lens_ptr[i];
      const int64_t other_pre_len = pre_lens_ptr[i];
      sum_extend_lens += other_seq_len - other_pre_len;
      sum_num_new_pages += ceil_div_i64(other_seq_len, page_size) -
          ceil_div_i64(other_pre_len, page_size);
    }

    const int64_t num_page_start_loc_self =
        ceil_div_i64(seq_len, page_size) - ceil_div_i64(pre_len, page_size);
    output_start_loc = sum_extend_lens - extend_len;
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self;
    ceil_pre_pages = ceil_div_i64(pre_len, page_size);
    last_loc = last_loc_ptr[pid];
  }

  __syncthreads();

  if (extend_len <= 0) {
    return;
  }

  const int64_t old_partial_end = ceil_pre_pages * page_size;
  for (int64_t offset = threadIdx.x; offset < extend_len; offset += blockDim.x) {
    const int64_t token_pos = pre_len + offset;
    int64_t loc;
    if (token_pos < old_partial_end) {
      loc = last_loc + 1 + offset;
    } else {
      const int64_t page_idx =
          new_page_start_loc + token_pos / page_size - ceil_pre_pages;
      const int64_t page_start = free_page_ptr[page_idx];
      loc = page_start * page_size + token_pos % page_size;
    }
    out_indices[output_start_loc + offset] = loc;
  }
}

}  // namespace

void lightop_hcu_alloc_extend_kernel(
    const at::Tensor pre_lens_ptr,
    const at::Tensor seq_lens_ptr,
    const at::Tensor last_loc_ptr,
    const at::Tensor free_page_ptr,
    at::Tensor out_indices,
    int64_t bs,
    int64_t page_size) {
  TORCH_CHECK(pre_lens_ptr.scalar_type() == at::kLong);
  TORCH_CHECK(seq_lens_ptr.scalar_type() == at::kLong);
  TORCH_CHECK(last_loc_ptr.scalar_type() == at::kLong);
  TORCH_CHECK(free_page_ptr.scalar_type() == at::kLong);
  TORCH_CHECK(out_indices.scalar_type() == at::kLong);
  TORCH_CHECK(pre_lens_ptr.is_contiguous());
  TORCH_CHECK(seq_lens_ptr.is_contiguous());
  TORCH_CHECK(last_loc_ptr.is_contiguous());
  TORCH_CHECK(free_page_ptr.is_contiguous());
  TORCH_CHECK(out_indices.is_contiguous());
  TORCH_CHECK(page_size > 0);
  TORCH_CHECK(bs >= 0);

  if (bs == 0 || out_indices.numel() == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(out_indices));
  constexpr int threads = 256;
  const dim3 grid(static_cast<unsigned int>(bs));
  const dim3 block(threads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lightop_alloc_extend_kernel<<<grid, block, 0, stream>>>(
      pre_lens_ptr.data_ptr<int64_t>(),
      seq_lens_ptr.data_ptr<int64_t>(),
      last_loc_ptr.data_ptr<int64_t>(),
      free_page_ptr.data_ptr<int64_t>(),
      out_indices.data_ptr<int64_t>(),
      bs,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace native
}  // namespace at
