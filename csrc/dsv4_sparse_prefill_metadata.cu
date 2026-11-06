// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include "dsv4_sparse_prefill_metadata.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdint>

namespace lightop::dsv4 {
namespace {
constexpr int threads = 256;
constexpr int vector_width = 4;

__device__ __forceinline__ int64_t wave_sum(int64_t value) {
    for (int delta = 32; delta; delta /= 2) value += __shfl_down(value, delta, 64);
    return __shfl(value, 0, 64);
}

template<bool PowerOfTwo>
__device__ __forceinline__ int64_t cp_length(
    int64_t start, int64_t length, int cp_size, int rank, int shift) {
    const int64_t phase = PowerOfTwo ? start & (cp_size - 1) : start % cp_size;
    const int64_t delta = rank - phase + cp_size;
    const int64_t first = start + (PowerOfTwo ? delta & (cp_size - 1) : delta % cp_size);
    const int64_t end = start + length;
    if (first >= end) return 0;
    return 1 + (PowerOfTwo ? (end - 1 - first) >> shift : (end - 1 - first) / cp_size);
}

template<bool PowerOfTwo, bool SingleRequest, int BlockThreads>
__global__ void cp_query_metadata(
    const int32_t* extend, const int32_t* starts, const int32_t* positions,
    int32_t* physical, int32_t* real, int32_t* sanitized,
    int64_t requests, int64_t queries, int64_t extend_stride,
    int64_t start_stride, int64_t position_stride, int cp_size, int rank, int cp_shift) {
    int64_t total;
    if constexpr (SingleRequest) {
        total = cp_length<PowerOfTwo>(starts[0], extend[0], cp_size, rank, cp_shift);
    } else {
        int64_t partial = 0;
        for (int64_t r = threadIdx.x & 63; r < requests; r += 64)
            partial += cp_length<PowerOfTwo>(starts[r * start_stride], extend[r * extend_stride], cp_size, rank, cp_shift);
        total = wave_sum(partial);
    }
    if (blockIdx.x == 0 && threadIdx.x < 64) {
        for (int64_t r = threadIdx.x; r < requests; r += 64) {
            const int64_t n = cp_length<PowerOfTwo>(starts[r * start_stride], extend[r * extend_stride], cp_size, rank, cp_shift);
            real[r] = static_cast<int32_t>(n);
            physical[r] = static_cast<int32_t>(n + (r + 1 == requests ? queries - total : 0));
        }
    }
    const int64_t first = (int64_t(blockIdx.x) * BlockThreads + threadIdx.x) * vector_width;
    if (first + vector_width <= queries && position_stride == 1 &&
        (reinterpret_cast<uintptr_t>(positions) & 15) == 0) {
        int4 values = reinterpret_cast<const int4*>(positions)[first / vector_width];
        values.x = first < total ? values.x : -1;
        values.y = first + 1 < total ? values.y : -1;
        values.z = first + 2 < total ? values.z : -1;
        values.w = first + 3 < total ? values.w : -1;
        reinterpret_cast<int4*>(sanitized)[first / vector_width] = values;
    } else {
        #pragma unroll
        for (int i = 0; i < vector_width; ++i) {
            const int64_t q = first + i;
            if (q < queries) sanitized[q] = q < total ? positions[q * position_stride] : -1;
        }
    }
}

template<typename Index, bool Pages>
__global__ void compact_token_ids(
    const Index* source, const int32_t* seq, int32_t* output, int32_t* bases,
    int64_t seq_stride, int64_t source_row_stride, int64_t source_col_stride,
    int64_t source_width, int64_t ratio, int64_t page_size, int page_shift) {
    const int64_t r = blockIdx.x;
    int64_t partial = 0;
    const int lane = threadIdx.x & 63;
    for (int64_t i = lane; i < r; i += 64) {
        const int64_t n = seq[i * seq_stride] >> (Pages ? 2 : 7);
        partial += n > 0 ? n : 1;
    }
    const int64_t offset = r ? wave_sum(partial) : 0;
    const int64_t n = seq[r * seq_stride] >> (Pages ? 2 : 7);
    const int64_t length = n > 0 ? n : 1;
    if (blockIdx.y == 0 && threadIdx.x == 0) bases[r] = static_cast<int32_t>(offset);
    const int64_t step = int64_t(gridDim.y) * threads * vector_width;
    for (int64_t first = (int64_t(blockIdx.y) * threads + threadIdx.x) * vector_width;
         first < length; first += step) {
        int32_t values[vector_width];
        if (Pages && page_shift >= 2) {
            const int64_t index = first >> page_shift;
            const int64_t id = index < source_width ? source[r * source_row_stride + index * source_col_stride] : -1;
            const int64_t physical = id * page_size + (first & (page_size - 1));
            #pragma unroll
            for (int i = 0; i < vector_width; ++i)
                values[i] = static_cast<int32_t>(physical + i > 0 ? physical + i : 0);
        } else {
            #pragma unroll
            for (int i = 0; i < vector_width; ++i) {
                const int64_t token = first + i;
                if (token < length) {
                    const int64_t index = Pages ? (page_shift >= 0 ? token >> page_shift : token / page_size) : token;
                    const int64_t id = index < source_width ? source[r * source_row_stride + index * source_col_stride] : -1;
                    const int64_t physical = Pages ? id * page_size + (page_shift >= 0 ? token & (page_size - 1) : token % page_size) : id;
                    values[i] = static_cast<int32_t>(physical > 0 ? physical : 0);
                }
            }
        }
        if (first + vector_width <= length && ((offset + first) & 3) == 0 &&
            (reinterpret_cast<uintptr_t>(output) & 15) == 0) {
            const int4 packed = make_int4(values[0], values[1], values[2], values[3]);
            reinterpret_cast<int4*>(output)[(offset + first) / vector_width] = packed;
        } else {
            #pragma unroll
            for (int i = 0; i < vector_width; ++i)
                if (first + i < length) output[offset + first + i] = values[i];
        }
    }
}

void vector_check(const at::Tensor& x, const at::Tensor& ref, int64_t size, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.device() == ref.device() && x.scalar_type() == at::kInt &&
                x.dim() == 1 && x.numel() == size, name, " must be an int32 vector on the same device");
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> build_cp_query_metadata(
    const at::Tensor& extend, const at::Tensor& starts,
    const at::Tensor& positions, int64_t cp_size, int64_t cp_rank) {
    TORCH_CHECK(extend.is_cuda() && extend.scalar_type() == at::kInt && extend.dim() == 1,
                "extend must be a device int32 vector");
    vector_check(starts, extend, extend.numel(), "starts");
    TORCH_CHECK(positions.dim() == 1, "positions must be one-dimensional");
    vector_check(positions, extend, positions.numel(), "positions");
    TORCH_CHECK(cp_size >= 1 && cp_size <= INT_MAX && cp_rank >= 0 && cp_rank < cp_size,
                "invalid CP size or rank");
    TORCH_CHECK(positions.numel() <= INT_MAX && extend.numel() <= INT_MAX,
                "query geometry exceeds int32 range");
    TORCH_CHECK(extend.numel() || positions.numel() == 0, "nonempty queries require requests");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(extend.device());
    auto real = at::empty({extend.numel()}, extend.options());
    auto physical = at::empty_like(real);
    auto sanitized = at::empty({positions.numel()}, positions.options());
    if (extend.numel() == 0) return {physical, real, sanitized};
    const int shift = (cp_size & (cp_size - 1)) == 0 ? __builtin_ctzll(cp_size) : -1;
    #define LAUNCH_CP(PowerOfTwo, SingleRequest, BlockThreads) \
        cp_query_metadata<PowerOfTwo, SingleRequest, BlockThreads><<< \
            std::max<int64_t>(1, (positions.numel() + BlockThreads * vector_width - 1) / (BlockThreads * vector_width)), \
            BlockThreads, 0, at::hip::getCurrentHIPStream()>>>( \
            extend.data_ptr<int32_t>(), starts.data_ptr<int32_t>(), positions.data_ptr<int32_t>(), \
            physical.data_ptr<int32_t>(), real.data_ptr<int32_t>(), sanitized.data_ptr<int32_t>(), \
            extend.numel(), positions.numel(), extend.stride(0), starts.stride(0), positions.stride(0), \
            cp_size, cp_rank, shift)
    #define DISPATCH_CP(BlockThreads) do { \
        if (shift >= 0) { \
            if (extend.numel() == 1) { LAUNCH_CP(true, true, BlockThreads); } \
            else { LAUNCH_CP(true, false, BlockThreads); } \
        } else { \
            if (extend.numel() == 1) { LAUNCH_CP(false, true, BlockThreads); } \
            else { LAUNCH_CP(false, false, BlockThreads); } \
        } \
    } while (false)
    if (positions.numel() < 65536) { DISPATCH_CP(64); }
    else { DISPATCH_CP(256); }
    #undef DISPATCH_CP
    #undef LAUNCH_CP
    TORCH_CHECK(hipGetLastError() == hipSuccess, "CP query metadata launch failed");
    return {physical, real, sanitized};
}

std::tuple<at::Tensor, at::Tensor> pack_compact_token_ids(
    const at::Tensor& source, const at::Tensor& seq,
    int64_t ratio, int64_t page_size, int64_t total,
    const c10::optional<at::Tensor>& out_ids, const c10::optional<at::Tensor>& out_base) {
    TORCH_CHECK(seq.is_cuda() && seq.scalar_type() == at::kInt && seq.dim() == 1,
                "seq must be a device int32 vector");
    TORCH_CHECK(source.device() == seq.device() && source.dim() == 2 && source.size(0) == seq.numel() &&
                (source.scalar_type() == at::kInt || source.scalar_type() == at::kLong),
                "source must be an int32/int64 request matrix on the same device");
    TORCH_CHECK((ratio == 4 || ratio == 128) && page_size > 0 && page_size <= INT_MAX,
                "invalid compressed-cache geometry");
    TORCH_CHECK(total >= seq.numel() && total <= INT_MAX && (seq.numel() || total == 0),
                "invalid compact output capacity");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(seq.device());
    auto ids = out_ids ? *out_ids : at::empty({total}, seq.options());
    auto bases = out_base ? *out_base : at::empty({seq.numel()}, seq.options());
    vector_check(ids, seq, total, "out_ids");vector_check(bases, seq, seq.numel(), "out_base");
    TORCH_CHECK(ids.is_contiguous() && bases.is_contiguous(), "output vectors must be contiguous");
    if (!seq.numel()) return {ids, bases};
    const int64_t per_worker = seq.numel() * threads * vector_width;
    const int workers = std::max<int64_t>(1, std::min<int64_t>(512, (total + per_worker - 1) / per_worker));
    const dim3 grid(seq.numel(), workers);
    const auto stream = at::hip::getCurrentHIPStream();
    const int page_shift = (page_size & (page_size - 1)) == 0 ? __builtin_ctzll(page_size) : -1;
    #define LAUNCH(Index, Pages) compact_token_ids<Index, Pages><<<grid, threads, 0, stream>>>( \
        source.data_ptr<Index>(), seq.data_ptr<int32_t>(), ids.data_ptr<int32_t>(), bases.data_ptr<int32_t>(), \
        seq.stride(0), source.stride(0), source.stride(1), source.size(1), ratio, page_size, page_shift)
    if (source.scalar_type() == at::kInt) {
        if (ratio == 4) { LAUNCH(int32_t, true); } else { LAUNCH(int32_t, false); }
    } else {
        if (ratio == 4) { LAUNCH(int64_t, true); } else { LAUNCH(int64_t, false); }
    }
    #undef LAUNCH
    TORCH_CHECK(hipGetLastError() == hipSuccess, "compact token-ID launch failed");
    return {ids, bases};
}
}  // namespace lightop::dsv4
