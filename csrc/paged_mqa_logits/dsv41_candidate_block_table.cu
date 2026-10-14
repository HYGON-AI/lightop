// Level one of the DSV4.1 two-level indexer in a single pass per row
// (replaces amax8_varlen + a block top-k + sort_candidate_blocks):
//
//   key[i]   = max(logits[b, 8 i : 8 i + 8]) for the n = ceil(len / 8) blocks,
//              the newest block +inf (always kept), a NaN maximum -inf
//   kept     = the topk blocks with the largest keys (ties: lowest block id)
//   out[b,:] = kept ascending as pool slots / 8, 0 past min(n, topk)
//   valid[b] = 8 (min(n, topk) - 1) + (len - 1) % 8 + 1  (0 for len 0)
//
// The kept set comes from an exact 4 x 8-bit radix select on order-preserving
// keys; the output is produced by one ordered compaction, so no sort is
// needed. Keys live in LDS for up to kLdsKeys blocks, else in an L2-resident
// workspace. A row with at most topk blocks gets the identity table.
//
// Logits of one row share a few exponents, so the top digits fall into one or
// two bins; each wave counts into its own histogram to keep LDS atomics from
// serializing across the whole CTA.
#include "paged_mqa_logits_fp4.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>

namespace lightop::deep_gemm {
namespace {
constexpr int kThreads = 1024;
constexpr int kWaves = kThreads / 64;
constexpr int kLdsKeys = 8192;
constexpr int kBlock = 8;
constexpr int kBinWaves = 256 / 64;  // waves holding one radix bin per lane

__device__ __forceinline__ unsigned order_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive block-wide prefix sum.
__device__ __forceinline__ int block_exclusive_scan(int v, int* s_wave) {
    const int lane = threadIdx.x % 64, wave = threadIdx.x / 64;
    int x = v;
#pragma unroll
    for (int d = 1; d < 64; d *= 2) {
        const int y = __shfl_up(x, d, 64);
        if (lane >= d) x += y;
    }
    if (lane == 63) s_wave[wave] = x;
    __syncthreads();
    if (threadIdx.x < 64) {
        int w = threadIdx.x < kWaves ? s_wave[threadIdx.x] : 0;
#pragma unroll
        for (int d = 1; d < 64; d *= 2) {
            const int y = __shfl_up(w, d, 64);
            if (threadIdx.x >= d) w += y;
        }
        if (threadIdx.x < kWaves) s_wave[threadIdx.x] = w;  // inclusive per wave
    }
    __syncthreads();
    const int before = wave == 0 ? 0 : s_wave[wave - 1];
    __syncthreads();
    return before + x - v;
}

template <bool OutputMask = false>
__global__ __launch_bounds__(kThreads) void candidate_block_table_kernel(
    const float* __restrict__ logits, int64_t logits_stride, const int* __restrict__ seq_lens,
    const int* __restrict__ page_table, int64_t pt_stride, int blocks_per_page, int width,
    int topk, unsigned* __restrict__ workspace, int64_t ws_stride,
    int* __restrict__ out, int64_t out_stride, int* __restrict__ valid_lens,
    bool* __restrict__ out_mask = nullptr) {
    __shared__ unsigned s_keys[kLdsKeys];
    __shared__ int s_hist[kWaves][256];
    __shared__ int s_wave[kWaves];
    __shared__ unsigned s_prefix;
    __shared__ int s_remain;

    const int b = blockIdx.x, tid = threadIdx.x;
    const int wave = tid / 64, lane = tid % 64;
    const int len = min(seq_lens[b], width);
    const int n = (len + kBlock - 1) / kBlock;
    const int kept = min(n, topk);
    if constexpr (!OutputMask) {
        if (tid == 0) valid_lens[b] = len > 0 ? kBlock * (kept - 1) + (len - 1) % kBlock + 1 : 0;
    }
    int* row_out = out + b * out_stride;
    auto phys = [&](int i) {
        if constexpr (OutputMask) return i;
        else return page_table[b * pt_stride + i / blocks_per_page] * blocks_per_page + i % blocks_per_page;
    };
    bool* row_mask = nullptr;
    if constexpr (OutputMask) row_mask = out_mask + static_cast<int64_t>(b) * width;

    if (n <= topk) {
        for (int i = tid; i < topk; i += kThreads) row_out[i] = i < n ? phys(i) : (OutputMask ? -1 : 0);
        if constexpr (OutputMask) {
            for (int i = tid; i < width; i += kThreads) row_mask[i] = i / kBlock < n;
        }
        return;
    }

    if constexpr (OutputMask) {
        for (int i = tid; i < width; i += kThreads) row_mask[i] = false;
    }
    unsigned* keys = n <= kLdsKeys ? s_keys : workspace + b * ws_stride;
    const float* row = logits + b * logits_stride;
    for (int i = tid; i < n; i += kThreads) {
        float m = INFINITY;
        if (i < n - 1) {
            const float4 x = *reinterpret_cast<const float4*>(row + i * kBlock);
            const float4 y = *reinterpret_cast<const float4*>(row + i * kBlock + 4);
            m = fmaxf(fmaxf(fmaxf(x.x, x.y), fmaxf(x.z, x.w)), fmaxf(fmaxf(y.x, y.y), fmaxf(y.z, y.w)));
            // A NaN key would order above +inf and could evict the newest block.
            if (m != m) m = -INFINITY;
        }
        keys[i] = order_key(m);
    }
    if (tid == 0) {
        s_prefix = 0;
        s_remain = topk;
    }
    __syncthreads();

    // Radix select: the topk-th largest key T, and how many keys equal to T
    // are still needed once every key above T is taken.
    unsigned mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < kWaves * 256; i += kThreads) s_hist[i / 256][i % 256] = 0;
        __syncthreads();
        // Read both before any thread of this round can overwrite them.
        const unsigned prefix = s_prefix;
        const int remain = s_remain;
        for (int i = tid; i < n; i += kThreads) {
            const unsigned k = keys[i];
            if ((k & mask) == prefix) atomicAdd(&s_hist[wave][(k >> shift) & 255u], 1);
        }
        __syncthreads();
        // Waves 0..3 own one bin per lane: count, then suffix = keys in bins >= bin.
        int count = 0, suffix = 0;
        if (tid < 256) {
#pragma unroll
            for (int w = 0; w < kWaves; ++w) count += s_hist[w][tid];
            suffix = count;
#pragma unroll
            for (int d = 1; d < 64; d *= 2) {
                const int y = __shfl_down(suffix, d, 64);
                if (lane + d < 64) suffix += y;
            }
            if (lane == 0) s_wave[wave] = suffix;
        }
        __syncthreads();
        if (tid < 256) {
            for (int w = wave + 1; w < kBinWaves; ++w) suffix += s_wave[w];
            const int above = suffix - count;
            if (suffix >= remain && above < remain) {
                s_prefix = prefix | (static_cast<unsigned>(tid) << shift);
                s_remain = remain - above;
            }
        }
        mask |= 255u << shift;
        __syncthreads();
    }
    const unsigned threshold = s_prefix;
    const int ties = s_remain;

    // Ordered compaction: contiguous chunks per thread keep block order.
    const int chunk = (n + kThreads - 1) / kThreads;
    const int lo = min(tid * chunk, n), hi = min(lo + chunk, n);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = keys[i];
        gt += k > threshold;
        eq += k == threshold;
    }
    const int gt_before = block_exclusive_scan(gt, s_wave);
    const int eq_before = block_exclusive_scan(eq, s_wave);
    int pos = gt_before + min(eq_before, ties);
    int eq_seen = eq_before;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = keys[i];
        if (k > threshold || (k == threshold && eq_seen++ < ties)) {
            row_out[pos++] = phys(i);
            if constexpr (OutputMask) {
#pragma unroll
                for (int j = 0; j < kBlock; ++j) {
                    if (i * kBlock + j < width) row_mask[i * kBlock + j] = true;
                }
            }
        }
    }
}
}  // namespace

void dsv41_candidate_block_table(
    const torch::Tensor& logits, const torch::Tensor& seq_lens, const torch::Tensor& page_table,
    int64_t page_size, int64_t block_size, torch::Tensor& out_phys_blocks, torch::Tensor& out_valid_lens) {
    TORCH_CHECK(block_size == kBlock, "dsv41_candidate_block_table: block_size must be 8");
    TORCH_CHECK(page_size > 0 && page_size % kBlock == 0, "dsv41_candidate_block_table: page_size must be a multiple of 8");
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 2 && logits.stride(1) == 1 &&
                logits.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
                "dsv41_candidate_block_table: logits must be FP32 [B,W] with 16-byte aligned rows");
    const int64_t batch = logits.size(0);
    TORCH_CHECK(seq_lens.device() == logits.device() && seq_lens.scalar_type() == at::kInt &&
                seq_lens.is_contiguous() && seq_lens.numel() == batch,
                "dsv41_candidate_block_table: seq_lens must be contiguous int32 [B]");
    TORCH_CHECK(page_table.device() == logits.device() && page_table.scalar_type() == at::kInt &&
                page_table.dim() == 2 && page_table.size(0) == batch && page_table.stride(1) == 1,
                "dsv41_candidate_block_table: page_table must be int32 [B,pages]");
    TORCH_CHECK(page_table.size(1) * page_size >= logits.size(1),
                "dsv41_candidate_block_table: page_table must cover the logits width");
    TORCH_CHECK(out_phys_blocks.device() == logits.device() && out_phys_blocks.scalar_type() == at::kInt &&
                out_phys_blocks.dim() == 2 && out_phys_blocks.size(0) == batch && out_phys_blocks.stride(1) == 1 &&
                out_phys_blocks.size(1) > 0,
                "dsv41_candidate_block_table: out_phys_blocks must be int32 [B,topk]");
    TORCH_CHECK(out_valid_lens.device() == logits.device() && out_valid_lens.scalar_type() == at::kInt &&
                out_valid_lens.is_contiguous() && out_valid_lens.numel() == batch,
                "dsv41_candidate_block_table: out_valid_lens must be contiguous int32 [B]");
    if (batch == 0) return;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(logits.device());
    const int64_t max_blocks = (logits.size(1) + kBlock - 1) / kBlock;
    torch::Tensor workspace;
    unsigned* ws = nullptr;
    int64_t ws_stride = 0;
    if (max_blocks > kLdsKeys) {
        workspace = torch::empty({batch, max_blocks}, logits.options().dtype(at::kInt));
        ws = reinterpret_cast<unsigned*>(workspace.data_ptr<int>());
        ws_stride = max_blocks;
    }
    auto stream = at::hip::getCurrentHIPStream();
    candidate_block_table_kernel<false><<<batch, kThreads, 0, stream>>>(
        logits.data_ptr<float>(), logits.stride(0), seq_lens.data_ptr<int>(), page_table.data_ptr<int>(),
        page_table.stride(0), static_cast<int>(page_size / kBlock), static_cast<int>(logits.size(1)),
        static_cast<int>(out_phys_blocks.size(1)), ws, ws_stride, out_phys_blocks.data_ptr<int>(),
        out_phys_blocks.stride(0), out_valid_lens.data_ptr<int>());
    TORCH_CHECK(hipGetLastError() == hipSuccess, "dsv41_candidate_block_table: launch failed");
}

void dsv41_select_candidate_blocks(
    const torch::Tensor& logits, const torch::Tensor& seq_lens, int64_t block_size,
    torch::Tensor& out_mask, torch::Tensor& out_block_ids) {
    TORCH_CHECK(block_size == kBlock, "dsv41_select_candidate_blocks: block_size must be 8");
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 2 && logits.stride(1) == 1 &&
                logits.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
                "dsv41_select_candidate_blocks: logits must be FP32 [B,W] with 16-byte aligned rows");
    const int64_t batch = logits.size(0), width = logits.size(1);
    TORCH_CHECK(seq_lens.device() == logits.device() && seq_lens.scalar_type() == at::kInt &&
                seq_lens.is_contiguous() && seq_lens.numel() == batch,
                "dsv41_select_candidate_blocks: seq_lens must be contiguous int32 [B]");
    TORCH_CHECK(out_mask.device() == logits.device() && out_mask.scalar_type() == at::kBool &&
                out_mask.sizes() == logits.sizes() && out_mask.is_contiguous(),
                "dsv41_select_candidate_blocks: out_mask must be contiguous bool [B,W]");
    TORCH_CHECK(out_block_ids.device() == logits.device() && out_block_ids.scalar_type() == at::kInt &&
                out_block_ids.dim() == 2 && out_block_ids.size(0) == batch && out_block_ids.is_contiguous() &&
                out_block_ids.size(1) > 0,
                "dsv41_select_candidate_blocks: out_block_ids must be contiguous int32 [B,topk]");
    if (batch == 0) return;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(logits.device());
    const int64_t max_blocks = (width + kBlock - 1) / kBlock;
    torch::Tensor workspace;
    unsigned* ws = nullptr;
    if (max_blocks > kLdsKeys) {
        workspace = torch::empty({batch, max_blocks}, logits.options().dtype(at::kInt));
        ws = reinterpret_cast<unsigned*>(workspace.data_ptr<int>());
    }
    auto stream = at::hip::getCurrentHIPStream();
    candidate_block_table_kernel<true><<<batch, kThreads, 0, stream>>>(
        logits.data_ptr<float>(), logits.stride(0), seq_lens.data_ptr<int>(), nullptr, 0, 1,
        static_cast<int>(width), static_cast<int>(out_block_ids.size(1)), ws, max_blocks,
        out_block_ids.data_ptr<int>(), out_block_ids.stride(0), nullptr, out_mask.data_ptr<bool>());
    TORCH_CHECK(hipGetLastError() == hipSuccess, "dsv41_select_candidate_blocks: launch failed");
}

}  // namespace lightop::deep_gemm
