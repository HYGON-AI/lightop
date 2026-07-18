"""Direct bindings for existing LightOp RoPE APIs."""

from ..fuse_rotary_positional_embedding import (
    apply_rotary_pos_emb_index_fused,
)

__all__ = ["apply_rotary_pos_emb_index_fused"]
