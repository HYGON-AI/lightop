#pragma once
#include <optional>
#include <torch/extension.h>

namespace lightop::deep_gemm {

/// \brief Fewest requests (batch / rows_per_request) at which paged_mqa_logits_fp4
///        pairs rows; below it the call runs as rows_per_request 1. Measured on gfx938
///        (64 CUs), H 32, n 6: fewer requests leave CUs idle and pairing lengthens CTAs.
inline constexpr int64_t kPagedMqaLogitsFp4PairMinRequests = 8;

/// Maximum number of route rows handled by the FP4 schedule planner.  The
/// planner keeps one prefix value per route in LDS; callers must chunk or use
/// the grid path above this bound.
inline constexpr int64_t kPagedMqaLogitsFp4MaxScheduleRows = 16384;

/// \brief DeepGEMM fp8_fp4_paged_mqa_logits contract (clean_logits = false) on HCU.
/// \param q [B,1,H,64] packed E2M1 (int8/uint8), H in {16,32,64}
/// \param q_sf [B,1,H] int32, four packed UE8M0 per-32 scales per head
/// \param kv_cache [pages,64,1,68] uint8 fused index-K pool, 16-byte aligned pages
/// \param weights [B,H] FP32 head weights
/// \param context_lens [B] int32 visible positions per row
/// \param block_table [B,blocks] int32 at table_block_size (64 dense, 8 candidate)
/// \param schedule_meta get_paged_mqa_logits_metadata(lens, 64, cus), or none; lens are
///        context_lens, or with rows_per_request > 1 the route lengths (see below)
/// \param max_context_len logits width; the row stride is rounded up to the split size
/// \param table_block_size positions per block_table entry, 8 or 64
/// \param out optional preallocated FP32 [B,>=max_context_len] with contiguous columns
/// \param rows_per_request n > 1 declares rows [b n, b n + n) one request's draft tokens
///        sharing block_table row b n (table_block_size 64 and H <= 32 only). With at
///        least kPagedMqaLogitsFp4PairMinRequests requests the kernel scores routes of
///        rows (b n + 2 j, b n + 2 j + 1), j < ceil(n / 2), in one KV pass each; fewer
///        run as n = 1. Logits are identical either way; the schedule is planned on
///        paged_mqa_logits_route_lens, which applies the same threshold
/// \return FP32 [B,max_context_len]; columns at or past context_lens[b] are unwritten
torch::Tensor paged_mqa_logits_fp4(
    const torch::Tensor& q, const torch::Tensor& q_sf, const torch::Tensor& kv_cache,
    const torch::Tensor& weights, const torch::Tensor& context_lens, const torch::Tensor& block_table,
    const std::optional<torch::Tensor>& schedule_meta, int64_t max_context_len,
    int64_t table_block_size = 64, std::optional<torch::Tensor> out = std::nullopt,
    int64_t rows_per_request = 1);

/// FP4 split-KV schedule planner.  It computes paired route lengths in the
/// planner kernel when rows_per_request enables paired execution.
torch::Tensor get_paged_mqa_logits_fp4_metadata(
    const torch::Tensor& context_lens, int64_t block_kv, int64_t num_sms,
    int64_t rows_per_request = 1);

/// \brief Level one of the DSV4.1 two-level indexer: keeps the topk best blocks of 8
///        by block max (newest block always kept), ascending, as pool slots / 8.
/// \param logits FP32 [B,W] dense index logits with 16-byte aligned rows
/// \param seq_lens [B] int32 visible positions per row
/// \param page_table [B,pages] int32 at page_size
/// \param page_size positions per page_table entry, a multiple of 8
/// \param block_size candidate block size, must be 8
/// \param out_phys_blocks [B,topk] int32, entries past min(n, topk) are zero
/// \param out_valid_lens [B] int32 sparse row length, 8 (kept - 1) + (len - 1) % 8 + 1
void dsv41_candidate_block_table(
    const torch::Tensor& logits, const torch::Tensor& seq_lens, const torch::Tensor& page_table,
    int64_t page_size, int64_t block_size, torch::Tensor& out_phys_blocks, torch::Tensor& out_valid_lens);

/// Logical candidate IDs with -1 padding and a dense bool mask, in one launch.
void dsv41_select_candidate_blocks(
    const torch::Tensor& logits, const torch::Tensor& seq_lens, int64_t block_size,
    torch::Tensor& out_mask, torch::Tensor& out_block_ids);

}  // namespace lightop::deep_gemm
