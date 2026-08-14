// Ported from custom_ops/custom_index_select_hip.hip for lightop P1.
// HIP runtime APIs rewritten to CUDA-style for lightop hipify build.

// Optimized index_select along dim=0 for contiguous 2D tensors (row gather).
//
// Workload shape (micro_perf): src[src_batch, dim], index[dst_batch] -> dst[dst_batch, dim]
//
// Strategy:
//   - 16B vectorized loads/stores (float4 / uint4)
//   - large dim: one block per output row + grid remap (L2 channel spread)
//   - small dim: multi-row per block, one warp (64) per row
//
// On BW1100 this recovers near-peak BW for large rows where ATen gather collapses.

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

#define CUDA_CHECK(cmd)                                                                 \
    do {                                                                               \
        cudaError_t error = (cmd);                                                      \
        if (error != cudaSuccess) {                                                     \
            TORCH_CHECK(false, "HIP error: ", cudaGetErrorString(error));               \
        }                                                                              \
    } while (0)

namespace {

constexpr unsigned kGridRemapMult = 4097u;
constexpr int kWavefront = 64;
constexpr int kLargeDimThreshold = 4096;

__device__ __forceinline__ int map_block(int bid, int bcount) {
    if (bcount <= 1 || bcount == 65536) {
        return bid;
    }
    return static_cast<int>(
        (static_cast<uint64_t>(bid) * kGridRemapMult) % static_cast<uint32_t>(bcount));
}

template <typename VecT, typename ScalarT>
__global__ void index_select_row_vec_kernel(
    const ScalarT* __restrict__ src,
    const int64_t* __restrict__ index,
    ScalarT* __restrict__ dst,
    int64_t dim,
    int64_t dst_batch,
    int vec_elems) {
    const int ob = map_block(static_cast<int>(blockIdx.x), static_cast<int>(dst_batch));
    if (ob >= dst_batch) {
        return;
    }

    const int64_t src_row = index[ob];
    const ScalarT* in = src + src_row * dim;
    ScalarT* out = dst + static_cast<int64_t>(ob) * dim;

    const int64_t nvec = dim / vec_elems;
    for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
        const int64_t i = vi * vec_elems;
        *reinterpret_cast<VecT*>(out + i) = *reinterpret_cast<const VecT*>(in + i);
    }
    for (int64_t i = nvec * vec_elems + threadIdx.x; i < dim; i += blockDim.x) {
        out[i] = in[i];
    }
}

template <typename VecT, typename ScalarT>
__global__ void index_select_multirow_vec_kernel(
    const ScalarT* __restrict__ src,
    const int64_t* __restrict__ index,
    ScalarT* __restrict__ dst,
    int64_t dim,
    int64_t dst_batch,
    int rows_per_block,
    int vec_elems) {
    const int base = static_cast<int>(blockIdx.x) * rows_per_block;
    const int warp = threadIdx.x / kWavefront;
    const int lane = threadIdx.x % kWavefront;
    const int warps = blockDim.x / kWavefront;

    for (int r = warp; r < rows_per_block; r += warps) {
        const int ob = base + r;
        if (ob >= dst_batch) {
            return;
        }
        const int64_t src_row = index[ob];
        const ScalarT* in = src + src_row * dim;
        ScalarT* out = dst + static_cast<int64_t>(ob) * dim;

        const int64_t nvec = dim / vec_elems;
        for (int64_t vi = lane; vi < nvec; vi += kWavefront) {
            const int64_t i = vi * vec_elems;
            *reinterpret_cast<VecT*>(out + i) = *reinterpret_cast<const VecT*>(in + i);
        }
        for (int64_t i = nvec * vec_elems + lane; i < dim; i += kWavefront) {
            out[i] = in[i];
        }
    }
}

template <typename VecT, typename ScalarT>
void launch_index_select(
    const ScalarT* src,
    const int64_t* index,
    ScalarT* out,
    int64_t dim,
    int64_t dst_batch,
    int vec_elems,
    cudaStream_t stream) {
    if (dim >= kLargeDimThreshold) {
        int threads = 256;
        if (dim >= 16384) {
            threads = 512;
        }
        if (dim >= 65536) {
            threads = 1024;
        }
        index_select_row_vec_kernel<VecT, ScalarT><<<dim3(dst_batch), dim3(threads), 0, stream>>>(
        src,
        index,
        out,
        dim,
        dst_batch,
        vec_elems);;
    } else {
        int rows_per_block = (dim <= 256) ? 8 : 4;
        const int blocks = static_cast<int>((dst_batch + rows_per_block - 1) / rows_per_block);
        index_select_multirow_vec_kernel<VecT, ScalarT><<<dim3(blocks), dim3(256), 0, stream>>>(
        src,
        index,
        out,
        dim,
        dst_batch,
        rows_per_block,
        vec_elems);;
    }
}

}  // namespace

void index_select_opt_impl(torch::Tensor src, torch::Tensor index, torch::Tensor out) {
    TORCH_CHECK(src.dim() == 2, "custom_index_select: src must be 2D");
    TORCH_CHECK(index.dim() == 1, "custom_index_select: index must be 1D");
    TORCH_CHECK(out.dim() == 2, "custom_index_select: out must be 2D");
    TORCH_CHECK(src.is_contiguous() && out.is_contiguous() && index.is_contiguous(),
                "custom_index_select: tensors must be contiguous");
    TORCH_CHECK(index.scalar_type() == at::kLong, "custom_index_select: index must be int64");

    const int64_t dim = src.size(1);
    const int64_t dst_batch = index.size(0);
    TORCH_CHECK(out.size(0) == dst_batch && out.size(1) == dim,
                "custom_index_select: out shape mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t* ip = index.data_ptr<int64_t>();

    if (src.scalar_type() == at::kFloat) {
        launch_index_select<float4, float>(
            src.data_ptr<float>(), ip, out.data_ptr<float>(), dim, dst_batch, /*vec=*/4, stream);
    } else if (src.scalar_type() == at::kBFloat16) {
        using BF = at::BFloat16;
        launch_index_select<uint4, BF>(
            src.data_ptr<BF>(), ip, out.data_ptr<BF>(), dim, dst_batch, /*vec=*/8, stream);
    } else if (src.scalar_type() == at::kHalf) {
        using H = at::Half;
        launch_index_select<uint4, H>(
            src.data_ptr<H>(), ip, out.data_ptr<H>(), dim, dst_batch, /*vec=*/8, stream);
    } else {
        TORCH_CHECK(false, "custom_index_select: only float16/bfloat16/float32 supported");
    }
    CUDA_CHECK(cudaGetLastError());
}


namespace at {
namespace native {

torch::Tensor index_select_opt(
    const torch::Tensor& src, int64_t dim, const torch::Tensor& index) {
  TORCH_CHECK(dim == 0, "index_select_opt: only dim=0 supported");
  TORCH_CHECK(src.is_cuda() && index.is_cuda(), "index_select_opt: CUDA/HIP tensors");
  TORCH_CHECK(src.dim() == 2 && index.dim() == 1, "index_select_opt: src 2D, index 1D");
  TORCH_CHECK(src.is_contiguous() && index.is_contiguous(), "index_select_opt: contiguous");
  TORCH_CHECK(index.scalar_type() == at::kLong, "index_select_opt: index int64");
  TORCH_CHECK(
      src.scalar_type() == at::kHalf || src.scalar_type() == at::kBFloat16 ||
          src.scalar_type() == at::kFloat,
      "index_select_opt: float16/bfloat16/float32 only");
  auto out = torch::empty({index.size(0), src.size(1)}, src.options());
  index_select_opt_impl(src, index, out);
  return out;
}

torch::Tensor index_select_opt_out(
    const torch::Tensor& src,
    int64_t dim,
    const torch::Tensor& index,
    torch::Tensor& out) {
  TORCH_CHECK(dim == 0, "index_select_opt: only dim=0 supported");
  index_select_opt_impl(src, index, out);
  return out;
}

}  // namespace native
}  // namespace at
