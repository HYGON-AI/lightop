"""Direct bindings for existing LightOp RoPE APIs."""

from .. import op as lightop_native
from ..fuse_rms_mrope_cuda import fuse_rms_mrope_cuda
from ..fuse_rotary_positional_embedding import (
    RotaryEmbedding,
    apply_rotary_pos_emb_index_fused,
    rotary_embedding_fuse,
)

rotary_embedding_deepseek_fuse = lightop_native.rotary_embedding_deepseek_fuse

__all__ = [
    "RotaryEmbedding",
    "apply_rotary_pos_emb_index_fused",
    "fuse_rms_mrope_cuda",
    "rotary_embedding_deepseek_fuse",
    "rotary_embedding_fuse",
]
