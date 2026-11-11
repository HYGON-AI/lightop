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

__global__ void build_swa_token_ids_kernel(
    const int32_t* seq, const int32_t* extend, const int32_t* req,
    const int32_t* table, const int64_t* full_to_swa,
    int32_t* ids, int32_t* first, int32_t* lens, int32_t* offsets,
    int64_t batch, int64_t window, int64_t seq_stride, int64_t extend_stride,
    int64_t req_stride, int64_t table_row_stride, int64_t table_col_stride,
    int64_t mapping_stride) {
    const int r = blockIdx.x;
    __shared__ int start, length, offset;
    if (threadIdx.x == 0) {
        int prefix = 0;
        for (int i = 0; i < r; ++i) {
            const int s = seq[i * seq_stride];
            const int64_t n = static_cast<int64_t>(extend[i * extend_stride]) + window - 1;
            prefix += s < n ? s : static_cast<int>(n);
        }
        offset = prefix;
        if (batch) {
            const int s = seq[r * seq_stride];
            const int64_t n = static_cast<int64_t>(extend[r * extend_stride]) + window - 1;
            length = s < n ? s : static_cast<int>(n);
            start = s - length;
            if (blockIdx.y == 0) {
                first[r] = start;
                lens[r] = length;
                offsets[r] = prefix;
                if (r + 1 == batch) offsets[batch] = prefix + length;
            }
        } else {
            length = 0;
            start = 0;
            offsets[0] = 0;
        }
    }
    __syncthreads();
    if (!length) return;
    const int64_t row = static_cast<int64_t>(req[r * req_stride]) * table_row_stride;
    for (int64_t i = static_cast<int64_t>(blockIdx.y) * kThreads + threadIdx.x;
         i < length; i += static_cast<int64_t>(gridDim.y) * kThreads) {
        const int64_t full_id = table[row + (start + i) * table_col_stride];
        ids[offset + i] = static_cast<int32_t>(full_to_swa[full_id * mapping_stride]);
    }
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_swa_token_ids(const torch::Tensor& seq, const torch::Tensor& extend,
                    const torch::Tensor& req, const torch::Tensor& table,
                    const torch::Tensor& full_to_swa, int64_t window, int64_t total) {
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
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(seq.device());
    auto ids = torch::empty({total}, seq.options());
    auto first = torch::empty({batch}, seq.options());
    auto lens = torch::empty({batch}, seq.options());
    auto offsets = torch::empty({batch + 1}, seq.options());
    // Each CTA computes its own prefix: no cross-CTA publication race.
    const int workers = std::max<int64_t>(1, std::min<int64_t>(128,
        (total + std::max<int64_t>(1, batch) * kThreads - 1) /
        (std::max<int64_t>(1, batch) * kThreads)));
    build_swa_token_ids_kernel<<<dim3(std::max<int64_t>(1, batch), workers), kThreads,
        0, at::hip::getCurrentHIPStream()>>>(seq.data_ptr<int32_t>(),
        extend.data_ptr<int32_t>(), req.data_ptr<int32_t>(), table.data_ptr<int32_t>(),
        full_to_swa.data_ptr<int64_t>(), ids.data_ptr<int32_t>(), first.data_ptr<int32_t>(),
        lens.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(), batch, window,
        seq.stride(0), extend.stride(0), req.stride(0), table.stride(0), table.stride(1),
        full_to_swa.stride(0));
    TORCH_CHECK(hipGetLastError() == hipSuccess, "build_swa_token_ids: kernel launch failed");
    return {ids, first, lens, offsets};
}
}  // namespace lightop::dsv41
