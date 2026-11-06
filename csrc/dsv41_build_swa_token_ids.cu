// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <tuple>

namespace lightop::dsv41 {
namespace {
constexpr int kThreads = 256;

__device__ __forceinline__ int64_t swa_wave_sum(int64_t x) {
    for (int delta = 32; delta; delta /= 2) x += __shfl_down(x, delta, 64);
    return __shfl(x, 0, 64);
}

__global__ void build_swa_token_ids_kernel(
    const int32_t* seq, const int32_t* extend, const int32_t* req,
    const int32_t* table, const int64_t* full_to_swa,
    int32_t* ids, int32_t* first, int32_t* lens, int32_t* offsets,
    int64_t batch, int64_t window, int64_t seq_stride, int64_t extend_stride,
    int64_t req_stride, int64_t table_row_stride, int64_t table_col_stride,
    int64_t mapping_stride, const int32_t* query_lens, int32_t* query_starts,
    int64_t query_stride) {
    if (!batch) {
        if (threadIdx.x == 0) {
            offsets[0] = 0;
            if (query_starts) query_starts[0] = 0;
        }
        return;
    }
    const int r = blockIdx.x;
    const int lane = threadIdx.x & 63;
    int64_t prefix_part = 0, query_part = 0;
    for (int i = lane; i < r; i += 64) {
        const int64_t n = static_cast<int64_t>(extend[i * extend_stride]) + window - 1;
        prefix_part += seq[i * seq_stride] < n ? seq[i * seq_stride] : n;
        if (query_lens) query_part += query_lens[i * query_stride];
    }
    const int64_t offset = r ? swa_wave_sum(prefix_part) : 0;
    const int64_t query_prefix = query_lens && r ? swa_wave_sum(query_part) : 0;
    const int64_t n = static_cast<int64_t>(extend[r * extend_stride]) + window - 1;
    const int64_t length = seq[r * seq_stride] < n ? seq[r * seq_stride] : n;
    const int64_t start = seq[r * seq_stride] - length;
    if (blockIdx.y == 0 && threadIdx.x == 0) {
        first[r] = static_cast<int32_t>(start);lens[r] = static_cast<int32_t>(length);
        offsets[r] = static_cast<int32_t>(offset);
        if (r + 1 == batch) offsets[batch] = static_cast<int32_t>(offset + length);
        if (query_starts) {
            query_starts[r] = static_cast<int32_t>(query_prefix);
            if (r + 1 == batch) query_starts[batch] = static_cast<int32_t>(query_prefix + query_lens[r * query_stride]);
        }
    }
    const int64_t row = static_cast<int64_t>(req[r * req_stride]) * table_row_stride;
    const int64_t tile_stride = static_cast<int64_t>(gridDim.y) * kThreads;
    for (int64_t first_token = static_cast<int64_t>(blockIdx.y) * kThreads + threadIdx.x;
         first_token < length; first_token += tile_stride * 4) {
        int64_t full[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t i = first_token + j * tile_stride;
            if (i < length) full[j] = table[row + (start + i) * table_col_stride];
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t i = first_token + j * tile_stride;
            if (i < length) ids[offset + i] = static_cast<int32_t>(full_to_swa[full[j] * mapping_stride]);
        }
    }
    return;
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_swa_token_ids(const torch::Tensor& seq, const torch::Tensor& extend,
                    const torch::Tensor& req, const torch::Tensor& table,
                    const torch::Tensor& full_to_swa, int64_t window, int64_t total,
                    const c10::optional<torch::Tensor>& query_lens,
                    const c10::optional<torch::Tensor>& query_starts) {
    TORCH_CHECK(seq.is_cuda() && seq.dim() == 1 && seq.scalar_type() == torch::kInt32,
                "build_swa_token_ids: seq must be a device int32 vector");
    const int64_t batch = seq.numel();
    for (const auto& input : {extend, req}) {
        TORCH_CHECK(input.device() == seq.device() && input.dim() == 1 &&
                    input.scalar_type() == torch::kInt32 && input.numel() == batch,
                    "build_swa_token_ids: request metadata must match seq");
    }
    TORCH_CHECK(table.device() == seq.device() && table.dim() == 2 &&
                table.scalar_type() == torch::kInt32,
                "build_swa_token_ids: table must be a device int32 matrix");
    TORCH_CHECK(full_to_swa.device() == seq.device() && full_to_swa.dim() == 1 &&
                full_to_swa.scalar_type() == torch::kInt64,
                "build_swa_token_ids: mapping must be a device int64 vector");
    TORCH_CHECK(window > 0 && total >= 0 && total <= INT32_MAX && batch <= INT32_MAX,
                "build_swa_token_ids: invalid window or output size");
    TORCH_CHECK(query_lens.has_value() == query_starts.has_value(),
                "query lengths and output starts must be provided together");
    if (query_lens) {
        TORCH_CHECK(query_lens->device() == seq.device() && query_lens->scalar_type() == at::kInt &&
                    query_lens->dim() == 1 && query_lens->numel() == batch,
                    "query lengths must be an int32 request vector on the same device");
        TORCH_CHECK(query_starts->device() == seq.device() && query_starts->scalar_type() == at::kInt &&
                    query_starts->dim() == 1 && query_starts->numel() == batch + 1 && query_starts->is_contiguous(),
                    "query starts must be a contiguous int32 output on the same device");
    }
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(seq.device());
    auto ids = torch::empty({total}, seq.options());
    auto first = torch::empty({batch}, seq.options());
    auto lens = torch::empty({batch}, seq.options());
    auto offsets = torch::empty({batch + 1}, seq.options());
    // Each CTA computes its own prefix: no cross-CTA publication race.
    const int workers = std::max<int64_t>(1, std::min<int64_t>(total > 65536 && total <= 262144 ? 128 : 512,
        (total + std::max<int64_t>(1, batch) * kThreads - 1) /
        (std::max<int64_t>(1, batch) * kThreads)));
    build_swa_token_ids_kernel<<<dim3(std::max<int64_t>(1, batch), workers), kThreads,
        0, at::hip::getCurrentHIPStream()>>>(seq.data_ptr<int32_t>(),
        extend.data_ptr<int32_t>(), req.data_ptr<int32_t>(), table.data_ptr<int32_t>(),
        full_to_swa.data_ptr<int64_t>(), ids.data_ptr<int32_t>(), first.data_ptr<int32_t>(),
        lens.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(), batch, window,
        seq.stride(0), extend.stride(0), req.stride(0), table.stride(0), table.stride(1),
        full_to_swa.stride(0), query_lens ? query_lens->data_ptr<int32_t>() : nullptr,
        query_starts ? query_starts->data_ptr<int32_t>() : nullptr,
        query_lens ? query_lens->stride(0) : 0);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "build_swa_token_ids: kernel launch failed");
    return {ids, first, lens, offsets};
}
}  // namespace lightop::dsv41
