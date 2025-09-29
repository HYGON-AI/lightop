import torch
from lightop import op
from typing import Optional

def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    factor: float = 1.0
) -> None:
    op.moe_sum(
        input,
        output,
        bias,
        expert_mask,
        factor
    )
def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    is_EP : bool = False 
) -> None:
    op.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map,
        expert_mask,
        num_local_tokens,
        is_EP
    )
    