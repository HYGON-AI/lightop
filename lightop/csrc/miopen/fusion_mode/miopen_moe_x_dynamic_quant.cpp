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
                                                     const Tensor& smooth,
                                                     const int expert_start_idx,
                                                     const int expert_end_idx,
                                                     const int num_shared_experts_per_rank,
                                                     const int num_tokens_offset,
                                                     const int num_tokens_per_rank)
{

    TensorDescriptor hidden_tensor(hidden);
    TensorDescriptor selected_experts_tensor(selected_experts);
    TensorDescriptor smooth_tensor(smooth);

    int64_t topk                 = selected_experts.sizes()[1];
    int64_t allocated_tokens     = num_tokens_per_rank * topk;
    int64_t x_dim                = num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens;
    int64_t num_experts_per_rank = smooth.sizes()[0] - num_shared_experts_per_rank;

    std::vector<int64_t> out_scatter_tokens_shape = {x_dim, hidden.sizes()[1]};

    Tensor out_scatter_tokens          = at::empty(out_scatter_tokens_shape, hidden.options());
    Tensor out_scatter_per_token_scale = at::empty(x_dim, hidden.options().dtype(at::kFloat));
    Tensor out_experts_token_count     = at::empty(num_experts_per_rank, hidden.options().dtype(at::kInt));
    Tensor out_token_offset            = at::empty({allocated_tokens, topk}, hidden.options().dtype(at::kInt));

    TensorDescriptor output_scatter_tokens(out_scatter_tokens);
    TensorDescriptor output_scatter_per_token_scale(out_scatter_per_token_scale);
    TensorDescriptor output_experts_token_count(out_experts_token_count);
    TensorDescriptor output_token_offset(out_token_offset);

    MIOPEN_CHECK(miopenMoeScatterDynamicQuantForward(getMiopenHandle_new(),
                                                     hidden_tensor.desc(),
                                                     hidden.data_ptr(),
                                                     selected_experts_tensor.desc(),
                                                     selected_experts.data_ptr(),
                                                     smooth_tensor.desc(),
                                                     smooth.data_ptr(),
                                                     output_scatter_tokens.desc(),
                                                     out_scatter_tokens.data_ptr(),
                                                     output_scatter_per_token_scale.desc(),
                                                     out_scatter_per_token_scale.data_ptr(),
                                                     output_experts_token_count.desc(),
                                                     out_experts_token_count.data_ptr(),
                                                     output_token_offset.desc(),
                                                     out_token_offset.data_ptr(),
                                                     expert_start_idx,
                                                     expert_end_idx,
                                                     num_shared_experts_per_rank,
                                                     num_tokens_offset,
                                                     num_tokens_per_rank));

    return {std::move(out_scatter_tokens),
            std::move(out_scatter_per_token_scale),
            std::move(out_experts_token_count),
            std::move(out_token_offset)};
};

std::vector<Tensor> miopen_moe_swiglu_dynamic_quant(const Tensor& scatter_tokens,
                                                    const Tensor& smooth_scale,
                                                    const Tensor& experts_tokens_count,
                                                    const int num_shared_experts_per_rank,
                                                    const int num_tokens_offset,
                                                    const int num_tokens_per_rank,
                                                    const float beta)
{

    TensorDescriptor scatter_tokens_tensor(scatter_tokens);
    TensorDescriptor smooth_scale_tensor(smooth_scale);
    TensorDescriptor experts_tokens_count_tensor(experts_tokens_count);

    int64_t x_dim = scatter_tokens.sizes()[0];
    int64_t y_dim = smooth_scale.sizes()[1];

    Tensor out_y               = at::empty({x_dim, y_dim}, scatter_tokens.options().dtype(at::kByte));
    Tensor out_per_token_scale = at::empty(x_dim, scatter_tokens.options().dtype(at::kFloat));

    TensorDescriptor output_y(out_y);
    TensorDescriptor output_per_token_scale(out_per_token_scale);

    MIOPEN_CHECK(miopenSwiGluForward(getMiopenHandle_new(),
                                     scatter_tokens_tensor.desc(),
                                     scatter_tokens.data_ptr(),
                                     smooth_scale_tensor.desc(),
                                     smooth_scale.data_ptr(),
                                     experts_tokens_count_tensor.desc(),
                                     experts_tokens_count.data_ptr(),
                                     num_shared_experts_per_rank,
                                     num_tokens_offset,
                                     num_tokens_per_rank,
                                     beta,
                                     output_per_token_scale.desc(),
                                     out_per_token_scale.data_ptr(),
                                     output_y.desc(),
                                     out_y.data_ptr()));

    return {std::move(out_y), std::move(out_per_token_scale)};
};

Tensor miopen_moe_gather(const Tensor& scatter_tokens,
                         const Tensor& token_offset,
                         const Tensor& expert_weights,
                         const int num_shared_experts_per_rank,
                         const int num_tokens_offset,
                         const int num_tokens_per_rank)
{
    TensorDescriptor scatter_tokens_tensor(scatter_tokens);
    TensorDescriptor token_offset_tesnor(token_offset);
    TensorDescriptor expert_weights_tensor(expert_weights);

    std::vector<int64_t> y_shape = {expert_weights.sizes()[0], scatter_tokens.sizes()[1]};

    Tensor out_y = at::empty(y_shape, scatter_tokens.options());
    TensorDescriptor output_y(out_y);

    MIOPEN_CHECK(miopenMoeGather(getMiopenHandle_new(),
                                 scatter_tokens_tensor.desc(),
                                 scatter_tokens.data_ptr(),
                                 token_offset_tesnor.desc(),
                                 token_offset.data_ptr(),
                                 expert_weights_tensor.desc(),
                                 expert_weights.data_ptr(),
                                 output_y.desc(),
                                 out_y.data_ptr(),
                                 num_shared_experts_per_rank,
                                 num_tokens_offset,
                                 num_tokens_per_rank));

    return std::move(out_y);
}

} // namespace native

} // namespace at
