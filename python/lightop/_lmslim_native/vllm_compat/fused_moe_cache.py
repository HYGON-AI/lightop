# SPDX-License-Identifier: Apache-2.0
# MoE workspace cache (adapted from vLLM-style fused_moe usage; see project NOTICE if required.)

from __future__ import annotations

import logging

import torch

from lightop import envs as lsenvs

logger = logging.getLogger(__name__)

moe_cache_singleton: torch.Tensor | None = None


def get_moe_cache(top_k_num,N,K,device,dtype):
    global moe_cache_singleton
    if moe_cache_singleton is None:
        moe_cache_singleton = torch.empty(lsenvs.LMSLIM_FUSED_MOE_CHUNK_SIZE * top_k_num * max(N, K), device=device, dtype=dtype)
        nbytes = moe_cache_singleton.element_size() * moe_cache_singleton.numel()
        logger.info("Initializing moe_cache_singleton shape: %s, memory: %.2f MB", tuple(moe_cache_singleton.shape), nbytes / (1024**2))
    return moe_cache_singleton