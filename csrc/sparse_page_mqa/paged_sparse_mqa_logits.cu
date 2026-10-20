// Shared gfx936/gfx938 sparse-MQA source tree.
// Both implementations are compiled into this translation unit. Architecture
// selection is a host property lookup; it adds no tensor copy or GPU launch.
#include "paged_sparse_mqa_logits_gfx936.inc"
#include "paged_sparse_mqa_logits_gfx938.inc"

namespace lightop::deep_gemm {
torch::Tensor fp8_fp4_paged_sparse_mqa_logits(
    const torch::Tensor& q, const torch::Tensor& q_scales,
    const torch::Tensor& fused_kv_cache, const torch::Tensor& weights,
    const torch::Tensor& metadata, int64_t num_max_sparse_blocks,
    int64_t sparse_block_kv,
    const std::optional<torch::Tensor>& context_lens,
    const std::optional<torch::Tensor>& sparse_kv_block_indices,
    const std::optional<torch::Tensor>& output,
    std::optional<int64_t> num_sms, bool is_prefill) {
    TORCH_CHECK(q.is_cuda(), "fp8_fp4_paged_sparse_mqa_logits: Q must be a HIP tensor");
    hipDeviceProp_t properties;
    TORCH_CHECK(hipGetDeviceProperties(&properties, q.get_device()) == hipSuccess,
                "fp8_fp4_paged_sparse_mqa_logits: cannot query HIP device");
    if (std::strncmp(properties.gcnArchName, "gfx938", 6) == 0) {
        return gfx938_detail::fp8_fp4_paged_sparse_mqa_logits(
            q, q_scales, fused_kv_cache, weights, metadata,
            num_max_sparse_blocks, sparse_block_kv, context_lens,
            sparse_kv_block_indices, output, num_sms, is_prefill);
    }
    // gfx936 retains BF16 MMAC and its adaptive packed-K path. gfx928 and
    // unsupported devices retain the original implementation's validation.
    return gfx936_detail::fp8_fp4_paged_sparse_mqa_logits(
        q, q_scales, fused_kv_cache, weights, metadata,
        num_max_sparse_blocks, sparse_block_kv, context_lens,
        sparse_kv_block_indices, output, num_sms, is_prefill);
}
torch::Tensor prefill_sparse_fp8_fp4_paged_mqa_logits(
    const torch::Tensor& q,const torch::Tensor& scales,const torch::Tensor& kv,
    const torch::Tensor& weights,const torch::Tensor& lens,const torch::Tensor& table,
    const torch::Tensor& indices,const torch::Tensor& candidates,
    int64_t capacity,int64_t block,std::optional<int64_t> sms,bool paired_rows,int threads) {
    TORCH_CHECK(q.is_cuda(),"prefill_sparse_fp8_fp4_paged_mqa_logits: HIP Q required");
    hipDeviceProp_t prop;TORCH_CHECK(hipGetDeviceProperties(&prop,q.get_device())==hipSuccess,"device query failed");
    const int64_t n=sms.value_or(prop.multiProcessorCount);
    auto metadata=torch::empty({get_paged_sparse_mqa_logits_metadata_bytes(q.size(0),capacity,block,
        q.scalar_type()==at::kChar || q.scalar_type()==at::kByte ? 640 : 512,n)},q.options().dtype(at::kByte));
    PrefillSparseMQAMetadataPlan plan;
    plan.context_lens=lens;plan.block_table=table;plan.indices=indices;plan.candidates=candidates;
    plan.paired_rows=paired_rows;plan.threads=threads;
    if(std::strncmp(prop.gcnArchName,"gfx938",6)==0)
        return gfx938_detail::fp8_fp4_paged_sparse_mqa_logits(q,scales,kv,weights,metadata,capacity,block,
            std::nullopt,std::nullopt,std::nullopt,sms,true,&plan);
    return gfx936_detail::fp8_fp4_paged_sparse_mqa_logits(q,scales,kv,weights,metadata,capacity,block,
        std::nullopt,std::nullopt,std::nullopt,sms,true,&plan);
}
}  // namespace lightop::deep_gemm
