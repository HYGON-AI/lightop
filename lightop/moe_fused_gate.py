import torch
import enum
import torch.nn as nn
from lightop import op

def moe_fused_gate(x, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor):
    return op.moe_fused_gate(x, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)