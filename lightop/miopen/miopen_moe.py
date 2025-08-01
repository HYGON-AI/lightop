from lightop import op


def miopen_moe_scatter_dynamic_quant(
    hidden,
    selected_experts,
    smooth,
    expert_start_idx,
    expert_end_idx,
    num_shared_experts_per_rank,
    num_tokens_offset,
    num_tokens_per_rank,
):
    """_summary_

    Args:
        hidden (Tensor): 输入张量，形状为 [num_tokens, hidden_size]
        selected_experts (Tensor): 输入张量，形状为 [num_tokens, topk]
        smooth (Tensor): 输入张量，形状为 [num_shared_experts_per_rank + num_experts_per_rank, hidden_size]
        expert_start_idx (int): expert 开始索引
        expert_end_idx (int): expert 结束索引
        num_shared_experts_per_rank (int): 共享的expert的数量, 默认为 0, 如果 shared_expert 可用, 请添加到 num_experts_per_rank
        num_tokens_offset (int): Token偏移, 如果shared_experts需要 num_tokens 进行拆分, 则这是当前排名的token_offset
        num_tokens_per_rank (int): 每维上的Token数量, 如果需要shared_experts num_tokens 在上拆分, 则这是当前排名的num_tokens

    Returns:
        vector<Tensor>: scatter_tokens, 形状为 [num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens, hidden_size]
                        scatter_per_token_scale, 形状为 [num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens]
                        experts_token_count, 形状为 [num_experts_per_rank]
                        token_offset, 形状为 [allocated_tokens, topk]
    """
    return op.miopen_moe_scatter_dynamic_quant(
        hidden,
        selected_experts,
        smooth,
        expert_start_idx,
        expert_end_idx,
        num_shared_experts_per_rank,
        num_tokens_offset,
        num_tokens_per_rank,
    )


def miopen_moe_swiglu_dynamic_quant(
    scatter_tokens,
    smooth_scale,
    experts_tokens_count,
    num_shared_experts_per_rank,
    num_tokens_offset,
    num_tokens_per_rank,
    beta,
):
    """_summary_

    Args:
        scatter_tokens (Tensor): 输入张量，形状为 [num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens, hidden_size * 2]
        smooth_scale (Tensor): 输入张量，形状为 [num_shared_experts_per_rank + num_experts_per_rank, hidden_size]
        experts_tokens_count (Tensor): 输入张量，形状为 [num_experts_per_rank]
        num_shared_experts_per_rank (int): 默认为 0, 如果 shared_expert 可用, 请添加到 num_experts_per_rank
        num_tokens_offset (int): 如果shared_experts需要 num_tokens 进行拆分, 则这是当前排名的token_offset
        num_tokens_per_rank (int): 如果需要shared_experts num_tokens 在上拆分, 则这是当前排名的num_tokens
        beta (float): _description_

    Returns:
        vector<Tensor>:: y, 形状为 [num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens, hidden_size]
                        per_token_scale, 形状为 [num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens]
    """
    return op.miopen_moe_swiglu_dynamic_quant(
        scatter_tokens,
        smooth_scale,
        experts_tokens_count,
        num_shared_experts_per_rank,
        num_tokens_offset,
        num_tokens_per_rank,
        beta,
    )


def miopen_moe_gather(
    scatter_tokens, token_offset, expert_weights, num_shared_experts_per_rank, num_tokens_offset, num_tokens_per_rank
):
    """_summary_

    Args:
        scatter_tokens (Tensor): 输入张量，形状为 [num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens, hidden_size]
        token_offset (Tensor): 输入张量，形状为 [allocated_tokens]
        expert_weights (Tensor): 输入张量，形状为 [num_tokens, topk]
        num_shared_experts_per_rank (int): 默认为 0, 如果 shared_expert 可用, 请添加到 num_experts_per_rank
        num_tokens_offset (int): 如果shared_experts需要 num_tokens 进行拆分, 则这是当前排名的token_offset
        num_tokens_per_rank (int): 如果需要shared_experts num_tokens 在上拆分, 则这是当前排名的num_tokens

    Returns:
        Tensor: y 形状为 [num_tokens, hidden_size]
    """
    return op.miopen_moe_gather(
        scatter_tokens,
        token_offset,
        expert_weights,
        num_shared_experts_per_rank,
        num_tokens_offset,
        num_tokens_per_rank,
    )
