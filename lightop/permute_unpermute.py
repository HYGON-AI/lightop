from torch.autograd import Function
import torch
from torch import Tensor

from lightop import op


# def groupgemm_permute(tokens, routing_map):
#     return op.groupgemm_permute(tokens, routing_map)

# def groupgemm_unpermute(permuted_tokens, sorted_indices, restore_shape, probs, routing_map, expert_m_count):
#     return op.groupgemm_unpermute(permuted_tokens, sorted_indices, restore_shape, probs, routing_map, expert_m_count)


def groupgemm_permute(tokens, routing_map):
    return op.groupgemm_permute(tokens, routing_map)

def groupgemm_unpermute(permuted_tokens, sorted_indices, restore_shape, probs, routing_map, expert_m_count):
    return op.groupgemm_unpermute(permuted_tokens, sorted_indices, restore_shape, probs, routing_map, expert_m_count)

def fuse_groupgemm_permute_quant(tokens, routing_map):
    num_rows = tokens.numel() // tokens.shape[-1]
    scales = torch.empty((num_rows * 8, 1),
                    device=tokens.device,
                    dtype=torch.float32)#"topK": 8
    return op.fuse_groupgemm_permute_quant(tokens, routing_map, scales)