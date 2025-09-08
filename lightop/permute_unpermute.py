from torch.autograd import Function
import torch
from torch import Tensor

from lightop import op


def groupgemm_permute(tokens, routing_map):
    return op.groupgemm_permute(tokens, routing_map)

def groupgemm_unpermute(permuted_tokens, sorted_indices, restore_shape, probs, routing_map, expert_m_count):
    return op.groupgemm_unpermute(permuted_tokens, sorted_indices, restore_shape, probs, routing_map, expert_m_count)