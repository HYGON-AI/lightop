// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "dsv41_mqa_prep.h"

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>

#include <climits>
#include <cstdint>
#include <type_traits>

namespace lightop::fp4_mqa {
namespace {

__global__ void query_lens_kernel(
    const int32_t* starts, const int32_t* lengths, int32_t* out,
    int64_t batch, int64_t cp_size, int64_t cp_rank) {
    const int64_t request = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (request >= batch) return;
    const int64_t start = starts[request];
    const int64_t end = start + lengths[request];
    const int64_t shift = cp_size - 1 - cp_rank;
    out[request] = static_cast<int32_t>((end + shift) / cp_size -
                                        (start + shift) / cp_size);
}

template <typename SeqLength, int Ratio>
__global__ void prefill_metadata_kernel(
    const SeqLength* seq_lens, const int32_t* q_lens,
    const int64_t* positions, const int64_t* req_pool_indices,
    const int32_t* req_to_token, int64_t request_stride,
    int32_t* ks, int32_t* ke, int32_t* compress_lens, int64_t* slots) {
    const int request = blockIdx.y;
    __shared__ int64_t query_start;
    __shared__ int64_t key_start;
    __shared__ int64_t query_length;
    __shared__ int64_t key_length;
    __shared__ int64_t pool_request;
    if (threadIdx.x == 0) {
        int64_t query_prefix = 0;
        int64_t key_prefix = 0;
        for (int previous = 0; previous < request; ++previous) {
            query_prefix += q_lens[previous];
            key_prefix += static_cast<int64_t>(seq_lens[previous]) / Ratio;
        }
        query_start = query_prefix;
        key_start = key_prefix;
        query_length = q_lens[request];
        key_length = static_cast<int64_t>(seq_lens[request]) / Ratio;
        pool_request = req_pool_indices[request];
    }
    __syncthreads();
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < query_length) {
        const int64_t row = query_start + index;
        const int64_t visible = (positions[row] + 1) / Ratio;
        ks[row] = static_cast<int32_t>(key_start);
        ke[row] = static_cast<int32_t>(key_start + visible);
        compress_lens[row] = static_cast<int32_t>(visible);
    }
    if (index < key_length) {
        const int64_t address = pool_request * request_stride + index * Ratio;
        slots[key_start + index] = static_cast<int64_t>(req_to_token[address]) / Ratio;
    }
}

__global__ void gather_index_k_fp4_kernel(
    const uint8_t* cache, const int64_t* slots,
    uint8_t* payload, uint32_t* scales,
    int64_t length, int64_t page_size, int64_t page_stride) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.x / 64;
    if (row >= length) return;
    const int column = threadIdx.x % 64;
    const int64_t slot = slots[row];
    const int64_t page = slot / page_size;
    const int64_t offset = slot % page_size;
    const int64_t base = page * page_stride;
    payload[row * 64 + column] = cache[base + offset * 64 + column];
    if (column == 0) {
        const uint8_t* source = cache + base + page_size * 64 + offset * 4;
        scales[row] = static_cast<uint32_t>(source[0]) |
                      (static_cast<uint32_t>(source[1]) << 8) |
                      (static_cast<uint32_t>(source[2]) << 16) |
                      (static_cast<uint32_t>(source[3]) << 24);
    }
}

}  // namespace

torch::Tensor dsv41_mqa_query_lens(
    const torch::Tensor& extend_start_loc, const torch::Tensor& extend_seq_lens,
    int64_t cp_size, int64_t cp_rank) {
    TORCH_CHECK(extend_start_loc.is_cuda() &&
                extend_start_loc.scalar_type() == at::kInt &&
                extend_start_loc.dim() == 1 && extend_start_loc.is_contiguous(),
                "extend_start_loc must be contiguous int32 [B] on HIP");
    TORCH_CHECK(extend_seq_lens.is_cuda() &&
                extend_seq_lens.device() == extend_start_loc.device() &&
                extend_seq_lens.scalar_type() == at::kInt &&
                extend_seq_lens.dim() == 1 &&
                extend_seq_lens.size(0) == extend_start_loc.size(0) &&
                extend_seq_lens.is_contiguous(),
                "extend_seq_lens must be contiguous int32 [B] on the same device");
    TORCH_CHECK(cp_size > 0 && cp_size <= INT_MAX && cp_rank >= 0 && cp_rank < cp_size,
                "CP size/rank must satisfy 0 <= rank < size <= INT_MAX");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(extend_start_loc.device());
    auto out = torch::empty_like(extend_seq_lens);
    const int64_t batch = extend_start_loc.size(0);
    TORCH_CHECK(batch <= INT_MAX, "batch exceeds query layout launch limits");
    if (batch == 0) return out;
    const auto stream = at::hip::getCurrentHIPStream();
    query_lens_kernel<<<(batch + 255) / 256, 256, 0, stream>>>(
        extend_start_loc.data_ptr<int32_t>(), extend_seq_lens.data_ptr<int32_t>(),
        out.data_ptr<int32_t>(), batch, cp_size, cp_rank);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "DSV41 query layout kernel launch failed");
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dsv41_mqa_prefill_metadata(
    const torch::Tensor& seq_lens, const torch::Tensor& q_lens,
    const torch::Tensor& positions, const torch::Tensor& req_pool_indices,
    const torch::Tensor& req_to_token, int64_t ratio,
    int64_t num_k_slots, int64_t max_request_work) {
    TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == at::kLong &&
                positions.dim() == 1 && positions.is_contiguous(),
                "positions must be contiguous int64 [T] on HIP");
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.device() == positions.device() &&
                (seq_lens.scalar_type() == at::kInt || seq_lens.scalar_type() == at::kLong) &&
                seq_lens.dim() == 1 && seq_lens.is_contiguous(),
                "seq_lens must be contiguous int32/int64 [B] on the same device");
    const int64_t batch = seq_lens.size(0);
    const int64_t tokens = positions.size(0);
    TORCH_CHECK(q_lens.is_cuda() && q_lens.device() == positions.device() &&
                q_lens.scalar_type() == at::kInt && q_lens.dim() == 1 &&
                q_lens.size(0) == batch && q_lens.is_contiguous(),
                "q_lens must be contiguous int32 [B] on the same device");
    TORCH_CHECK(req_pool_indices.is_cuda() &&
                req_pool_indices.device() == positions.device() &&
                req_pool_indices.scalar_type() == at::kLong &&
                req_pool_indices.dim() == 1 && req_pool_indices.size(0) == batch &&
                req_pool_indices.is_contiguous(),
                "req_pool_indices must be contiguous int64 [B] on the same device");
    TORCH_CHECK(req_to_token.is_cuda() && req_to_token.device() == positions.device() &&
                req_to_token.scalar_type() == at::kInt && req_to_token.dim() == 2 &&
                req_to_token.is_contiguous(),
                "req_to_token must be contiguous int32 [capacity,width] on the same device");
    TORCH_CHECK(ratio == 1 || ratio == 2, "ratio must be 1 or 2");
    TORCH_CHECK(batch <= 65535 && tokens <= INT_MAX &&
                num_k_slots >= 0 && num_k_slots <= INT_MAX / 64,
                "shape exceeds metadata or MQA buffer offset limits");
    TORCH_CHECK(max_request_work >= 0 && max_request_work <= INT_MAX &&
                (max_request_work > 0 || (tokens == 0 && num_k_slots == 0)) &&
                (batch > 0 || (tokens == 0 && num_k_slots == 0)),
                "max_request_work/batch must cover nonempty outputs");
    TORCH_CHECK(num_k_slots == 0 || req_to_token.size(1) > 0,
                "req_to_token width must be positive for nonempty slots");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(positions.device());
    auto ks = torch::empty({tokens}, positions.options().dtype(at::kInt));
    auto ke = torch::empty_like(ks);
    auto compress_lens = torch::empty_like(ks);
    auto slots = torch::empty({num_k_slots}, positions.options().dtype(at::kLong));
    if (batch == 0 || max_request_work == 0) return {ks, ke, compress_lens, slots};
    const auto stream = at::hip::getCurrentHIPStream();
    const dim3 grid((max_request_work + 255) / 256, batch);
    const auto launch = [&](auto* lengths, auto ratio_value) {
        constexpr int kRatio = decltype(ratio_value)::value;
        using SeqLength = std::remove_pointer_t<decltype(lengths)>;
        prefill_metadata_kernel<SeqLength, kRatio><<<grid, 256, 0, stream>>>(
            lengths, q_lens.data_ptr<int32_t>(), positions.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int64_t>(), req_to_token.data_ptr<int32_t>(),
            req_to_token.stride(0), ks.data_ptr<int32_t>(), ke.data_ptr<int32_t>(),
            compress_lens.data_ptr<int32_t>(), slots.data_ptr<int64_t>());
    };
    if (seq_lens.scalar_type() == at::kInt) {
        if (ratio == 1) launch(seq_lens.data_ptr<int32_t>(), std::integral_constant<int, 1>{});
        else launch(seq_lens.data_ptr<int32_t>(), std::integral_constant<int, 2>{});
    } else {
        if (ratio == 1) launch(seq_lens.data_ptr<int64_t>(), std::integral_constant<int, 1>{});
        else launch(seq_lens.data_ptr<int64_t>(), std::integral_constant<int, 2>{});
    }
    TORCH_CHECK(hipGetLastError() == hipSuccess, "DSV41 prefill metadata kernel launch failed");
    return {ks, ke, compress_lens, slots};
}

std::tuple<torch::Tensor, torch::Tensor> dsv41_gather_index_k_fp4(
    const torch::Tensor& cache, const torch::Tensor& slots, int64_t page_size) {
    TORCH_CHECK(page_size > 0 && page_size <= INT_MAX / 68,
                "page_size must be positive and fit page byte offsets");
    TORCH_CHECK(cache.is_cuda() && cache.scalar_type() == at::kByte &&
                cache.dim() == 2 && cache.size(1) == page_size * 68 &&
                cache.stride(1) == 1 && cache.stride(0) >= cache.size(1),
                "cache must be uint8 [pages,page_size*68] with contiguous page rows on HIP");
    TORCH_CHECK(slots.is_cuda() && slots.device() == cache.device() &&
                slots.scalar_type() == at::kLong && slots.dim() == 1 &&
                slots.is_contiguous(),
                "slots must be contiguous int64 [L] on the cache device");
    const int64_t length = slots.size(0);
    TORCH_CHECK(length <= INT_MAX / 64, "length exceeds MQA buffer offset limits");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(cache.device());
    auto payload = torch::empty({length, 64}, cache.options().dtype(at::kChar));
    auto scales = torch::empty({length}, cache.options().dtype(at::kInt));
    if (length == 0) return {payload, scales};
    const auto stream = at::hip::getCurrentHIPStream();
    gather_index_k_fp4_kernel<<<(length + 3) / 4, 256, 0, stream>>>(
        cache.data_ptr<uint8_t>(), slots.data_ptr<int64_t>(),
        reinterpret_cast<uint8_t*>(payload.data_ptr<int8_t>()),
        reinterpret_cast<uint32_t*>(scales.data_ptr<int32_t>()),
        length, page_size, cache.stride(0));
    TORCH_CHECK(hipGetLastError() == hipSuccess, "DSV41 FP4 gather kernel launch failed");
    return {payload, scales};
}

}  // namespace lightop::fp4_mqa
