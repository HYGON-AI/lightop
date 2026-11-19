// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Ported from custom_ops/custom_reduceminmax_hip.hip for lightop P1.
// HIP runtime APIs rewritten to CUDA-style for lightop hipify build.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#define CUDA_CHECK(cmd)                                                                          \
    do                                                                                          \
    {                                                                                           \
        cudaError_t error = (cmd);                                                               \
        if(error != cudaSuccess)                                                                 \
        {                                                                                       \
            TORCH_CHECK(false,                                                                  \
                        "HIP error: ",                                                         \
                        cudaGetErrorString(error),                                               \
                        " at ",                                                                \
                        __FILE__,                                                               \
                        ":",                                                                   \
                        __LINE__);                                                              \
        }                                                                                       \
    } while(0)

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x) __builtin_expect(!!(x), 1)

// Scatter concurrent CTAs across rows so power-of-two row strides do not thrash
// the same L2 sets. Use an odd multiplier that is NOT congruent to 1 mod common
// batch sizes (4097 ≡ 1 mod 1024, so it was a no-op for M=1024 workloads).
#define GRID_REMAP_MULT 16381

__device__ __forceinline__ int map_row_idx(int block_idx, int block_count)
{
    if (block_count <= 1)
        return block_idx;
    const uint32_t bc = static_cast<uint32_t>(block_count);
    return static_cast<int>((static_cast<uint64_t>(block_idx) * GRID_REMAP_MULT) % bc);
}

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<__half>
{
    using Type = __half;
    static __device__ __forceinline__ float to_float(Type t) { return __half2float(t); }
    static __device__ __forceinline__ Type from_float(float t) { return __float2half(t); }
};

template <>
struct TypeTraits<__hip_bfloat16>
{
    using Type = __hip_bfloat16;
    static __device__ __forceinline__ float to_float(Type t) { return __bfloat162float(t); }
    static __device__ __forceinline__ Type from_float(float t) { return __float2bfloat16(t); }
};

template <>
struct TypeTraits<float>
{
    using Type = float;
    static __device__ __forceinline__ float to_float(Type t) { return t; }
    static __device__ __forceinline__ Type from_float(float t) { return t; }
};

struct ValIdx
{
    float val;
    int idx;
};

template <bool IsMax>
__device__ __forceinline__ ValIdx better(ValIdx a, ValIdx b)
{
    const bool a_wins_val = IsMax ? (a.val > b.val) : (a.val < b.val);
    const bool tie = (a.val == b.val);
    const bool take_a = a_wins_val || (tie && (a.idx <= b.idx));
    ValIdx out;
    out.val = take_a ? a.val : b.val;
    out.idx = take_a ? a.idx : b.idx;
    return out;
}

template <bool IsMax>
__device__ __forceinline__ void update_best(float f, int idx, float& best_val, int& best_idx)
{
    const bool better_val = IsMax ? (f > best_val) : (f < best_val);
    const bool tie = (f == best_val) && (idx < best_idx);
    const bool take = better_val || tie;
    best_val = take ? f : best_val;
    best_idx = take ? idx : best_idx;
}

template <bool IsMax>
__device__ __forceinline__ ValIdx warpReduceValIdx(ValIdx v, int reduce_size = 64)
{
    for (int offset = reduce_size / 2; offset > 0; offset /= 2)
    {
        ValIdx other;
        other.val = __shfl_xor(v.val, offset);
        other.idx = __shfl_xor(v.idx, offset);
        v = better<IsMax>(v, other);
    }
    return v;
}

// 16B vector loads: 8x fp16/bf16 or 4x fp32.
using LoadType = uint4;

template <typename T>
constexpr int kElementsPerLoad = static_cast<int>(sizeof(LoadType) / sizeof(T));

template <typename T, bool IsMax>
__device__ __forceinline__ void accumulate_vector_minmax(
    const LoadType& src, int base_idx, float& best_val, int& best_idx)
{
    const T* data = reinterpret_cast<const T*>(&src);
    constexpr int E = kElementsPerLoad<T>;

    ValIdx local{TypeTraits<T>::to_float(data[0]), base_idx};
#pragma unroll
    for (int i = 1; i < E; ++i)
        local = better<IsMax>(local, ValIdx{TypeTraits<T>::to_float(data[i]), base_idx + i});
    update_best<IsMax>(local.val, local.idx, best_val, best_idx);
}

// Allow 2 CTAs/CU (was 1): better latency hiding for many concurrent rows.
template <typename T, bool IsMax>
__global__ __launch_bounds__(512, 2) void reduce_minmax_row_kernel(
    const T* __restrict__ input,
    T* __restrict__ values,
    int64_t* __restrict__ indices,
    int M,
    int N)
{
    int row = map_row_idx(blockIdx.x, M);
    if (row >= M)
        return;

    int tid = threadIdx.x;
    int lane = tid % 64;
    int warp = tid / 64;
    int num_warps = blockDim.x / 64;

    const T* row_ptr = input + static_cast<int64_t>(row) * N;

    float best_val = IsMax ? -INFINITY : INFINITY;
    int best_idx = INT_MAX;

    if (N <= 0)
    {
        if (tid == 0)
        {
            values[row] = TypeTraits<T>::from_float(IsMax ? -INFINITY : INFINITY);
            indices[row] = 0;
        }
        return;
    }

    constexpr int E = kElementsPerLoad<T>;
    const int step = blockDim.x * E;
    const int vec_end = (N / E) * E;

    // Stagger the starting column across rows so concurrent CTAs do not march
    // in lockstep over power-of-two strides (L2 set conflicts on BW1100).
    int start_offset = 0;
    if (N >= 512)
    {
        start_offset = (N / 64) * (row % 64);
        start_offset = (start_offset / E) * E;
        if (start_offset >= vec_end)
            start_offset = 0;
    }

    // Phase 1: [start_offset, vec_end)
    for (int off = start_offset + tid * E; off < vec_end; off += step)
    {
        LoadType v = *reinterpret_cast<const LoadType*>(row_ptr + off);
        accumulate_vector_minmax<T, IsMax>(v, off, best_val, best_idx);
    }
    // Phase 2: wrap [0, start_offset)
    if (start_offset > 0)
    {
        for (int off = tid * E; off < start_offset; off += step)
        {
            LoadType v = *reinterpret_cast<const LoadType*>(row_ptr + off);
            accumulate_vector_minmax<T, IsMax>(v, off, best_val, best_idx);
        }
    }

    for (int i = vec_end + tid; i < N; i += blockDim.x)
        update_best<IsMax>(TypeTraits<T>::to_float(row_ptr[i]), i, best_val, best_idx);

    ValIdx local{best_val, best_idx};
    local = warpReduceValIdx<IsMax>(local);

    __shared__ float shared_vals[16];
    __shared__ int shared_idxs[16];

    if (lane == 0)
    {
        shared_vals[warp] = local.val;
        shared_idxs[warp] = local.idx;
    }

    __syncthreads();

    if (warp == 0)
    {
        ValIdx block_best;
        if (tid < num_warps)
        {
            block_best.val = shared_vals[tid];
            block_best.idx = shared_idxs[tid];
        }
        else
        {
            block_best.val = IsMax ? -INFINITY : INFINITY;
            block_best.idx = INT_MAX;
        }

        block_best = warpReduceValIdx<IsMax>(block_best, num_warps);

        if (tid == 0)
        {
            values[row] = TypeTraits<T>::from_float(block_best.val);
            indices[row] = static_cast<int64_t>(block_best.idx);
        }
    }
}


template <bool IsMax>
static void reduce_minmax_opt_impl(
    torch::Tensor input, torch::Tensor values, torch::Tensor indices)
{
    TORCH_CHECK(input.is_cuda(), "custom_ops: input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dim() == 2, "custom_ops: expected 2D tensor [M, N]");
    TORCH_CHECK(input.is_contiguous(), "custom_ops: input must be contiguous");
    TORCH_CHECK(values.is_cuda() && indices.is_cuda(), "custom_ops: outputs must be CUDA/HIP");
    TORCH_CHECK(values.is_contiguous() && indices.is_contiguous(),
                "custom_ops: values/indices must be contiguous");
    TORCH_CHECK(indices.scalar_type() == at::kLong, "custom_ops: indices must be int64");
    TORCH_CHECK(values.numel() >= input.size(0) && indices.numel() >= input.size(0),
                "custom_ops: values/indices size mismatch");
    TORCH_CHECK(values.scalar_type() == input.scalar_type(), "custom_ops: values dtype mismatch");

    int M = input.size(0);
    int N = input.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto launch_kernel = [&](int block_size) {
        dim3 block(block_size);
        dim3 grid(M);

        if (input.scalar_type() == at::kHalf)
        {
            using scalar_t_device = __half;
            reduce_minmax_row_kernel<scalar_t_device, IsMax><<<grid, block, 0, stream>>>(
        reinterpret_cast<const scalar_t_device*>(input.data_ptr<at::Half>()),
        reinterpret_cast<scalar_t_device*>(values.data_ptr<at::Half>()),
        indices.data_ptr<int64_t>(),
        M,
        N);;
        }
        else if (input.scalar_type() == at::kBFloat16)
        {
            using scalar_t_device = __hip_bfloat16;
            reduce_minmax_row_kernel<scalar_t_device, IsMax><<<grid, block, 0, stream>>>(
        reinterpret_cast<const scalar_t_device*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<scalar_t_device*>(values.data_ptr<at::BFloat16>()),
        indices.data_ptr<int64_t>(),
        M,
        N);;
        }
        else if (input.scalar_type() == at::kFloat)
        {
            // Also covers micro_perf dtype "tfloat32" (torch.float32 storage).
            using scalar_t_device = float;
            reduce_minmax_row_kernel<scalar_t_device, IsMax><<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        M,
        N);;
        }
        else
        {
            TORCH_CHECK(false, "custom_ops: unsupported dtype (fp16/bf16/fp32 only)");
        }
    };

    // Smoother ladder than 128<->512 at N=16384 (that cliff showed up as mem_bw jumps).
    int block_size;
    if (N <= 2048)
        block_size = 64;
    else if (N <= 8192)
        block_size = 128;
    else if (N <= 32768)
        block_size = 256;
    else
        block_size = 512;

    if (const char* env = std::getenv("REDUCEMINMAX_BLOCK_SIZE"))
    {
        int bs = std::atoi(env);
        if (bs == 64 || bs == 128 || bs == 256 || bs == 512 || bs == 1024)
            block_size = bs;
    }
    launch_kernel(block_size);
}

void reduce_max_opt_impl(torch::Tensor input, torch::Tensor values, torch::Tensor indices)
{
    reduce_minmax_opt_impl<true>(input, values, indices);
}

void reduce_min_opt_impl(torch::Tensor input, torch::Tensor values, torch::Tensor indices)
{
    reduce_minmax_opt_impl<false>(input, values, indices);
}


namespace at {
namespace native {

static void _check_reduce_minmax_io(
    const torch::Tensor& input, const torch::Tensor& values, const torch::Tensor& indices) {
  TORCH_CHECK(input.is_cuda(), "reduce_minmax_opt: input must be CUDA/HIP");
  TORCH_CHECK(input.dim() == 2, "reduce_minmax_opt: expected 2D [M,N]");
  TORCH_CHECK(input.is_contiguous(), "reduce_minmax_opt: input must be contiguous");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
          input.scalar_type() == at::kFloat,
      "reduce_minmax_opt: only float16/bfloat16/float32 supported");
  TORCH_CHECK(values.is_cuda() && indices.is_cuda(), "reduce_minmax_opt: values/indices CUDA/HIP");
  TORCH_CHECK(values.is_contiguous() && indices.is_contiguous(),
              "reduce_minmax_opt: values/indices contiguous");
  TORCH_CHECK(values.scalar_type() == input.scalar_type(), "reduce_minmax_opt: values dtype");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "reduce_minmax_opt: indices must be int64");
  TORCH_CHECK(values.numel() >= input.size(0) && indices.numel() >= input.size(0),
              "reduce_minmax_opt: values/indices length >= M");
}

std::tuple<torch::Tensor, torch::Tensor> reduce_max_opt(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.is_contiguous(),
              "reduce_max_opt: contiguous 2D CUDA/HIP required");
  auto values = torch::empty({input.size(0)}, input.options());
  auto indices = torch::empty({input.size(0)}, input.options().dtype(at::kLong));
  reduce_max_opt_impl(input, values, indices);
  return {values, indices};
}

std::tuple<torch::Tensor, torch::Tensor> reduce_max_opt_out(
    const torch::Tensor& input, torch::Tensor& values, torch::Tensor& indices) {
  _check_reduce_minmax_io(input, values, indices);
  reduce_max_opt_impl(input, values, indices);
  return {values, indices};
}

std::tuple<torch::Tensor, torch::Tensor> reduce_min_opt(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.is_contiguous(),
              "reduce_min_opt: contiguous 2D CUDA/HIP required");
  auto values = torch::empty({input.size(0)}, input.options());
  auto indices = torch::empty({input.size(0)}, input.options().dtype(at::kLong));
  reduce_min_opt_impl(input, values, indices);
  return {values, indices};
}

std::tuple<torch::Tensor, torch::Tensor> reduce_min_opt_out(
    const torch::Tensor& input, torch::Tensor& values, torch::Tensor& indices) {
  _check_reduce_minmax_io(input, values, indices);
  reduce_min_opt_impl(input, values, indices);
  return {values, indices};
}

}  // namespace native
}  // namespace at
