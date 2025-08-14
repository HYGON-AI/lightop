import torch
from torch import Tensor
from lightop import op
from typing import Optional


def miopen_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_msk_: Optional[Tensor] = None,
    droprate=0.0,
    is_causal=False,
    scale: Optional[float] = None,
    enable_gqa=False,
):
    """_summary_

    Args:
        query (_type_): 输入张量, 形状为 [B, H, S, D]
        key (_type_): 输入张量, 形状为 [B, H_k, S_k, D]
        value (_type_): 输入张量, 形状为 [B, H_k, S_k, D_v]
        attn_msk_ (Optional[Tensor]): _description_
        droprate (float, optional): 丢弃率. 默认为 0.0.
        is_causal (bool, optional): _description_. 默认为 False.
        scale (Optional[float], optional): 缩放因子. 默认为 None.
        enable_gqa (bool, optional): _description_. 默认为 False.

    Returns:
        Tensor: 输出Tensor, 形状为 [B, H, S, D_v]
    """
    return op.miopen_scaled_dot_product_attention(query, key, value, attn_msk_, droprate, is_causal, scale, enable_gqa)
