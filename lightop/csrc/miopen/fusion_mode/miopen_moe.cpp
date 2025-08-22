#include "miopen/miopen-wrapper.h"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <ATen/ATen.h>
#include "miopen/Descriptors.h"
#include <iostream>
namespace at {
namespace native {
std::vector<Tensor> miopen_moe_scatter_dynamic_quant(const Tensor& hidden,
                                                     const Tensor& selected_experts,
                                                     const Tensor& moe_weights,
                                                     const Tensor& smooth,
                                                     Tensor& out_scatter_tokens,
                                                     Tensor& out_scatter_per_token_scale,
                                                     Tensor& out_scatter_tokens_offset,
                                                     Tensor& out_experts_token_count,
                                                     Tensor& out_experts_token_start,
                                                     const int expert_start_idx,
                                                     const int expert_end_idx,
                                                     const int num_shared_experts,
                                                     const int num_tokens_offset,
                                                     const int num_tokens_per_rank,
                                                     const int num_tokens)
{
    if(hidden.dim() != 2 || selected_experts.dim() != 2 || moe_weights.dim() != 2)
        MIOPEN_THROW("Moe_Scatter_Dynamic_Quant: Input Tensor's dimension Does Not Match.");
    if(out_scatter_tokens.dim() != 2 || out_scatter_per_token_scale.dim() != 1 ||
       out_scatter_tokens_offset.dim() != 1 || out_experts_token_count.dim() != 1 || out_experts_token_start.dim() != 1)
        MIOPEN_THROW("Moe_Scatter_Dynamic_Quant: Output Tensor's dimension Does Not Match.");

    TensorDescriptor hidden_tensor(hidden);
    TensorDescriptor selected_experts_tensor(selected_experts);
    TensorDescriptor moe_weights_tensor(moe_weights);
    TensorDescriptor smooth_tensor(smooth);
    TensorDescriptor output_scatter_tokens(out_scatter_tokens);
    TensorDescriptor output_scatter_per_token_scale(out_scatter_per_token_scale);
    TensorDescriptor output_scatter_token_offset(out_scatter_tokens_offset);
    TensorDescriptor output_experts_token_count(out_experts_token_count);
    TensorDescriptor output_token_start(out_experts_token_start);

    // std::vector<Tensor> result =
    //     at::where((selected_experts >= expert_start_idx) & (selected_experts < expert_end_idx));
    // std::vector<Tensor> unique_results;
    // for(const auto& tensor : result)
    // {
    //     unique_results.push_back(std::get<0>(at::_unique(tensor)));
    // }
    // Tensor associateToken = unique_results[0].to(at::kLong);
    // TensorDescriptor associate_token_tensor(associateToken);

    size_t workspace_size = 0;
    miopenHandle_t handle = getMiopenHandle_new();
    MIOPEN_CHECK(miopenGetMoeScatterDynamicQuantForwardWorkspaceSize(handle,
                                                                     hidden_tensor.desc(),
                                                                     selected_experts_tensor.desc(),
                                                                     moe_weights_tensor.desc(),
                                                                     smooth_tensor.desc(),
                                                                     selected_experts_tensor.desc(),
                                                                     output_scatter_tokens.desc(),
                                                                     output_scatter_per_token_scale.desc(),
                                                                     output_experts_token_count.desc(),
                                                                     output_token_start.desc(),
                                                                     output_scatter_token_offset.desc(),
                                                                     expert_start_idx,
                                                                     expert_end_idx,
                                                                     num_shared_experts,
                                                                     num_tokens_offset,
                                                                     num_tokens_per_rank,
                                                                     num_tokens,
                                                                     &workspace_size));

    Tensor workspace_tensor;
    if(workspace_size > 0)
    {
        workspace_tensor = at::empty(workspace_size, hidden.options());
    }

    MIOPEN_CHECK(miopenMoeScatterDynamicQuantForward(handle,
                                                     hidden_tensor.desc(),
                                                     hidden.data_ptr(),
                                                     selected_experts_tensor.desc(),
                                                     selected_experts.data_ptr(),
                                                     moe_weights_tensor.desc(),
                                                     moe_weights.data_ptr(),
                                                     smooth_tensor.desc(),
                                                     smooth.data_ptr(),
                                                     selected_experts_tensor.desc(),
                                                     nullptr,
                                                     output_scatter_tokens.desc(),
                                                     out_scatter_tokens.data_ptr(),
                                                     output_scatter_per_token_scale.desc(),
                                                     out_scatter_per_token_scale.data_ptr(),
                                                     output_experts_token_count.desc(),
                                                     out_experts_token_count.data_ptr(),
                                                     output_token_start.desc(),
                                                     out_experts_token_start.data_ptr(),
                                                     output_scatter_token_offset.desc(),
                                                     out_scatter_tokens_offset.data_ptr(),
                                                     expert_start_idx,
                                                     expert_end_idx,
                                                     num_shared_experts,
                                                     num_tokens_offset,
                                                     num_tokens_per_rank,
                                                     num_tokens,
                                                     workspace_tensor.defined() ? workspace_tensor.data_ptr() : nullptr,
                                                     workspace_size));

    return {std::move(out_scatter_tokens),
            std::move(out_scatter_per_token_scale),
            std::move(out_scatter_tokens_offset),
            std::move(out_experts_token_count),
            std::move(out_experts_token_start)};
};

std::vector<Tensor> miopen_moe_swiglu_dynamic_quant(const Tensor& scatter_tokens,
                                                    const Tensor& smooth_scale,
                                                    const Tensor& experts_tokens_count,
                                                    const Tensor& exports_tokens_start,
                                                    const float beta = 1.0)
{

    if(scatter_tokens.dim() != 2 || smooth_scale.dim() != 2 || experts_tokens_count.dim() != 1 ||
       exports_tokens_start.dim() != 1)
        MIOPEN_THROW("Moe_Swiglu_Dynamic_Quant: Input Tensor's dimension Does Not Match.");

    TensorDescriptor scatter_tokens_tensor(scatter_tokens);
    TensorDescriptor smooth_scale_tensor(smooth_scale);
    TensorDescriptor experts_tokens_count_tensor(experts_tokens_count);
    TensorDescriptor exports_tokens_start_tensor(exports_tokens_start);

    int64_t token_nums = scatter_tokens.sizes()[0];

    Tensor out_y = at::empty({token_nums, smooth_scale.sizes()[1]}, scatter_tokens.options().dtype(at::kChar));
    Tensor out_per_token_scale = at::empty(token_nums, scatter_tokens.options().dtype(at::kFloat));

    TensorDescriptor output_y(out_y);
    TensorDescriptor output_per_token_scale(out_per_token_scale);

    MIOPEN_CHECK(miopenSwiGluForward(getMiopenHandle_new(),
                                     scatter_tokens_tensor.desc(),
                                     scatter_tokens.data_ptr(),
                                     smooth_scale_tensor.desc(),
                                     smooth_scale.data_ptr(),
                                     experts_tokens_count_tensor.desc(),
                                     experts_tokens_count.data_ptr(),
                                     exports_tokens_start_tensor.desc(),
                                     exports_tokens_start.data_ptr(),
                                     output_per_token_scale.desc(),
                                     out_per_token_scale.data_ptr(),
                                     output_y.desc(),
                                     out_y.data_ptr(),
                                     beta));

    return {std::move(out_y), std::move(out_per_token_scale)};
};

Tensor miopen_moe_gather(const Tensor& scatter_tokens,
                         const Tensor& token_offset,
                         const int real_scatter_tokens,
                         const int num_tokens,
                         const int hidden_size,
                         const int topk,
                         const int num_shared_experts)
{
    if(scatter_tokens.dim() != 2 || token_offset.dim() != 1)
        MIOPEN_THROW("Moe_Gather: Input Tensor's dimension Does Not Match.");

    TensorDescriptor scatter_tokens_tensor(scatter_tokens);
    TensorDescriptor token_offset_tesnor(token_offset);

    Tensor out_y = at::empty({num_tokens, hidden_size}, scatter_tokens.options());
    TensorDescriptor output_y(out_y);

    size_t workspace_size = 0;
    miopenHandle_t handle = getMiopenHandle_new();
    MIOPEN_CHECK(miopenGetMoeGatherWorkspaceSize(handle,
                                                 scatter_tokens_tensor.desc(),
                                                 scatter_tokens.data_ptr(),
                                                 token_offset_tesnor.desc(),
                                                 token_offset.data_ptr(),
                                                 output_y.desc(),
                                                 out_y.data_ptr(),
                                                 real_scatter_tokens,
                                                 num_tokens,
                                                 hidden_size,
                                                 topk,
                                                 num_shared_experts,
                                                 &workspace_size));

    Tensor workspace_tensor;
    if(workspace_size > 0)
    {
        workspace_tensor = at::empty(workspace_size, scatter_tokens.options());
    }

    MIOPEN_CHECK(miopenMoeGather(handle,
                                 scatter_tokens_tensor.desc(),
                                 scatter_tokens.data_ptr(),
                                 token_offset_tesnor.desc(),
                                 token_offset.data_ptr(),
                                 output_y.desc(),
                                 out_y.data_ptr(),
                                 real_scatter_tokens,
                                 num_tokens,
                                 hidden_size,
                                 topk,
                                 num_shared_experts,
                                 workspace_tensor.defined() ? workspace_tensor.data_ptr() : nullptr,
                                 workspace_size));

    return std::move(out_y);
}

} // namespace native

} // namespace at
