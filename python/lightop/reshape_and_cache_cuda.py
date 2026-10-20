# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import enum
#from lightop import op
from . import op
from typing import Tuple

def reshape_and_cache_cuda(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    op.reshape_and_cache_cuda(key, value, key_cache, 
                                                  value_cache, slot_mapping, 
                                                  kv_cache_dtype, k_scale, v_scale)