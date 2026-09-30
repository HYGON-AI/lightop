# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
#from lightop import op
from . import op
from typing import Optional, Tuple

def ceil_div(a, b):
    return (a + b - 1) // b

def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    factor: float = 1.0,
    expect_m: int = -1,
) -> None:
    op.moe_sum(
        input,
        output,
        bias,
        expert_mask,
        num_local_tokens,
        factor,
        expect_m,
    )

def moe_align_block_size(
     topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
    num_token: Optional[int] = None,
    ep_size: int = 8,
    is_EP : bool  = False,
    local_num_tokens: Optional[torch.Tensor]=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.
    - pad_sorted_ids: A flag indicating whether the sorted_token_ids length
      should be padded to a multiple of block_size,

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    local_num_experts = num_experts // ep_size if is_EP else num_experts
    if num_token:
        if num_token < block_size:
            max_num_tokens_padded = min(topk_ids.numel() * block_size, topk_ids.numel() + local_num_experts * (block_size - 1))
        else:
            max_num_tokens_padded = topk_ids.numel() + local_num_experts * (block_size - 1)
        sorted_ids = torch.full((max_num_tokens_padded,), fill_value=topk_ids.numel(), dtype=torch.int32, device=topk_ids.device)
    else:
        max_num_tokens_padded = topk_ids.numel() + local_num_experts * (block_size - 1)
        if pad_sorted_ids:
            max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
        sorted_ids = torch.empty((max_num_tokens_padded, ),
                                 dtype=torch.int32,
                                 device=topk_ids.device)
        #融到算子里了
        # sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = ceil_div(max_num_tokens_padded, block_size)
    # Expert ids must be zeroed out to prevent index out of bounds error while
    # mapping global expert ids to local expert ids in expert parallelism.
    if expert_map is not None:
        expert_ids = torch.zeros((max_num_m_blocks, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
    else:
        expert_ids = torch.empty((max_num_m_blocks, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
    num_tokens_post_pad = torch.zeros((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)   
    op.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                                 expert_ids, num_tokens_post_pad,expert_map,expert_mask,local_num_tokens,is_EP);
    
    return sorted_ids, expert_ids, num_tokens_post_pad
