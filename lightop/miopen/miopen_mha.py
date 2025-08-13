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
        query (Tensor ): _description_
        key (_type_): _description_
        value (_type_): _description_
        attn_msk_ (Optional[Tensor]): _description_
        droprate (float, optional): _description_. Defaults to 0.0.
        is_causal (bool, optional): _description_. Defaults to False.
        scale (Optional[float], optional): _description_. Defaults to None.
        enable_gqa (bool, optional): _description_. Defaults to False.

    Returns:
        Tensor: 结果张量, 形状为 [batch, h, s, d]
    """
    return op.miopen_scaled_dot_product_attention(query, key, value, attn_msk_, droprate, is_causal, scale, enable_gqa)
