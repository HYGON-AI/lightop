# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
#from lightop import op
from . import op
import torch.nn.functional as F


def rotary_embedding_fuse(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Args:
        positions: Token position indices. Shape ``[num_tokens]`` or
            ``[batch_size, seq_len]`` and dtype ``torch.int64``.
        query: Query tensor. Its leading dimensions must match ``positions``;
            the remaining dimensions may be ``[num_heads * head_size]`` or
            ``[num_heads, head_size]``.
        key: Optional key tensor with the same token/batch dimensions as
            ``query``. Its remaining dimensions may be
            ``[num_kv_heads * head_size]`` or ``[num_kv_heads, head_size]``.
        head_size: Size of each query/key head. Sizes ``64``, ``72``, ``128``
            and ``256`` use optimized kernels when their dispatch conditions
            are met; other sizes use the native fallback kernel.
        cos_sin_cache: Combined cosine/sine cache with shape
            ``[max_position, rot_dim]``. It must have the same dtype and device
            as ``query``; ``rot_dim`` must not exceed ``head_size``.
        is_neox: ``True`` for GPT-NeoX split-half rotary layout, or ``False``
         for GPT-J interleaved rotary layout.
    """
    if positions.dtype != torch.int64:
        raise TypeError(f"positions must have dtype torch.int64, got {positions.dtype}")
    if positions.device != query.device:
        raise ValueError("positions and query must be on the same device")
    if cos_sin_cache.dim() != 2:
        raise ValueError(
            "cos_sin_cache must have shape [max_position, rot_dim], "
            f"got {tuple(cos_sin_cache.shape)}"
        )
    if cos_sin_cache.size(1) > head_size:
        raise ValueError(
            f"rot_dim ({cos_sin_cache.size(1)}) must not exceed head_size ({head_size})"
        )
    if query.dtype != cos_sin_cache.dtype or query.device != cos_sin_cache.device:
        raise ValueError("query and cos_sin_cache must have the same dtype and device")
    if key is not None and (key.dtype != query.dtype or key.device != query.device):
        raise ValueError("key and query must have the same dtype and device")

    op.rotary_embedding_fuse(
        positions, query, key, head_size, cos_sin_cache, is_neox
    )
    return query, key


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


class RotaryPositionalEmbeddingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, cos, sin):
        q_ = q.contiguous()
        cos_ = cos.contiguous()
        sin_ = sin.contiguous()
        output = op.rotary_positional_embedding_forward(*q.shape, q_, cos_, sin_)
        ctx.save_for_backward(cos, sin)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        cos_, sin_ = ctx.saved_tensors
        grad_q = op.rotary_positional_embedding_backward(*grad_output.shape, grad_output, cos_, sin_)

        return grad_q, None, None


def apply_rotary_pos_emb_index_fused(q, k, cos, sin, position_id):
    # position_id: [sq, b], q: [sq, b * np, hn] -> [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    sq, b, np = position_id.size(0), position_id.size(1), q.size(1) // position_id.size(1)
    np_k = k.size(1)// position_id.size(1)
    q, k = q.view(sq, b, np, -1), k.view(sq, b, np_k, -1)
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q = RotaryPositionalEmbeddingFunction.apply(q, cos, sin)
    k = RotaryPositionalEmbeddingFunction.apply(k, cos, sin)
    return q.view(sq, b * np, -1), k.view(sq, b * np_k, -1)
