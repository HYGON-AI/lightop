#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>

namespace at {
namespace native {

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
  T val[N];
  __device__ inline T &operator[](int i) { return val[i]; }
  __device__ inline const T &operator[](int i) const { return val[i]; }
};

template <typename scalar_t, int HC, int VEC_SIZE>
__global__ void dpskv4_hc_head_fused_kernel(
    const float* __restrict__ mixes,
    const scalar_t* __restrict__ x,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    scalar_t* __restrict__ out,
    int num_tokens,
    int hidden_size,
    int blocks_per_token,
    float hc_eps) {
  using Vec = AlignedVector<scalar_t, VEC_SIZE>;

  const int block_id = blockIdx.x;
  const int token_idx = block_id / blocks_per_token;
  const int col_block = block_id - token_idx * blocks_per_token;
  const int col = (col_block * blockDim.x + threadIdx.x) * VEC_SIZE;
  if (token_idx >= num_tokens || col >= hidden_size) {
    return;
  }

  float pre[HC];
  const float scale = hc_scale[0];
#pragma unroll
  for (int h = 0; h < HC; ++h) {
    const float z = mixes[token_idx * HC + h] * scale + hc_base[h];
    pre[h] = 1.0f / (1.0f + expf(-z)) + hc_eps;
  }

  float acc[VEC_SIZE];
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    acc[i] = 0.0f;
  }

#pragma unroll
  for (int h = 0; h < HC; ++h) {
    const scalar_t* src =
        x + (static_cast<size_t>(token_idx) * HC + h) * hidden_size + col;
    Vec x_vec = *reinterpret_cast<const Vec*>(src);
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      acc[i] += pre[h] * static_cast<float>(x_vec.val[i]);
    }
  }

  Vec out_vec;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    out_vec.val[i] = static_cast<scalar_t>(acc[i]);
  }

  scalar_t* dst = out + static_cast<size_t>(token_idx) * hidden_size + col;
  *reinterpret_cast<Vec*>(dst) = out_vec;
}

torch::Tensor dpskv4_hc_head_fused(
    torch::Tensor mixes,
    torch::Tensor x,
    torch::Tensor hc_scale,
    torch::Tensor hc_base,
    double hc_eps) {
  TORCH_CHECK(mixes.is_cuda() && x.is_cuda() && hc_scale.is_cuda() &&
                  hc_base.is_cuda(),
              "dpskv4_hc_head_fused expects CUDA tensors");
  TORCH_CHECK(mixes.scalar_type() == at::ScalarType::Float,
              "dpskv4_hc_head_fused expects fp32 mixes");
  TORCH_CHECK(hc_scale.scalar_type() == at::ScalarType::Float &&
                  hc_base.scalar_type() == at::ScalarType::Float,
              "dpskv4_hc_head_fused expects fp32 hc_scale/hc_base");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half ||
                  x.scalar_type() == at::ScalarType::BFloat16 ||
                  x.scalar_type() == at::ScalarType::Float,
              "dpskv4_hc_head_fused expects fp16, bf16, or fp32 x");
  TORCH_CHECK(x.dim() == 3, "dpskv4_hc_head_fused expects x to be 3D");
  TORCH_CHECK(mixes.dim() == 2,
              "dpskv4_hc_head_fused expects mixes to be 2D");
  TORCH_CHECK(x.is_contiguous() && mixes.is_contiguous() &&
                  hc_scale.is_contiguous() && hc_base.is_contiguous(),
              "dpskv4_hc_head_fused expects contiguous tensors");

  const int num_tokens = x.size(0);
  const int hc = x.size(1);
  const int hidden_size = x.size(2);
  TORCH_CHECK(hc == 4, "dpskv4_hc_head_fused currently supports hc_mult=4");
  TORCH_CHECK(mixes.size(0) == num_tokens && mixes.size(1) == hc,
              "dpskv4_hc_head_fused got mismatched mixes shape");
  TORCH_CHECK(hc_scale.numel() == 1,
              "dpskv4_hc_head_fused expects hc_scale.numel() == 1");
  TORCH_CHECK(hc_base.numel() == hc,
              "dpskv4_hc_head_fused expects hc_base.numel() == hc_mult");

  auto out = torch::empty({num_tokens, hidden_size}, x.options());
  if (num_tokens == 0 || hidden_size == 0) {
    return out;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "dpskv4_hc_head_fused", [&] {
        constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
        TORCH_CHECK(hidden_size % VEC_SIZE == 0,
                    "dpskv4_hc_head_fused hidden_size must be divisible by "
                    "16 bytes");
        const int threads = 256;
        const int cols_per_block = threads * VEC_SIZE;
        const int blocks_per_token =
            (hidden_size + cols_per_block - 1) / cols_per_block;
        dim3 grid(static_cast<uint32_t>(
            static_cast<int64_t>(blocks_per_token) * num_tokens));
        dim3 block(threads);
        dpskv4_hc_head_fused_kernel<scalar_t, 4, VEC_SIZE>
            <<<grid, block, 0, stream>>>(
                mixes.data_ptr<float>(), x.data_ptr<scalar_t>(),
                hc_scale.data_ptr<float>(), hc_base.data_ptr<float>(),
                out.data_ptr<scalar_t>(), num_tokens, hidden_size,
                blocks_per_token, static_cast<float>(hc_eps));
      });

  return out;
}

} // namespace native
} // namespace at
