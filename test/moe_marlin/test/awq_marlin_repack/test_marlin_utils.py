# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional, Tuple
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # → lightop/test
from common.scalar_type import scalar_types
from common.quant_utils import awq_pack, quantize_weights

from marlin_quant import get_weight_perm, marlin_weights

import lightop
import lightop.gemm_ops

MARLIN_K_CHUNKS = [128]
MARLIN_N_CHUNKS = [64, 256]
# MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
MARLIN_SUPPORTED_GROUP_SIZES = [64, 128]
MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (1, 7, 5),
    (13, 17, 67),
    (26, 37, 13),
    (67, 13, 11),
    (257, 13, 11),
    (658, 13, 11),
]

# 将 NumPy 数组写入 Excel 文件
def write_numpy_to_excel(data, file_path):
    # 将 NumPy 数组转换为 Pandas DataFrame
    df = pd.DataFrame(data)
    
    # 将 DataFrame 写入 Excel 文件
    df.to_excel(file_path, index=False, header=False)


def rand_data(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")

@pytest.mark.parametrize("size_k", [7168, 256]) # 128
@pytest.mark.parametrize("size_n", [512, 7168]) # 128
@pytest.mark.parametrize("group_size", [64, 128]) # -1, 32, 64, 128
def test_awq_marlin_repack_uint4(size_k, size_n, group_size):

    quant_type = scalar_types.uint4

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Create input
    b_weight = rand_data((size_k, size_n)) # [size_k, size_n] fp16

    # Quantize
    w_ref, q_w, s, zp = quantize_weights(b_weight,
                                         quant_type,
                                         group_size, 
                                         zero_points=True) # 和python接口同样的 量化函数

    # Pack to AWQ format
    q_w_awq = awq_pack(q_w, quant_type.size_bits, size_k, size_n)

    # Pack to Marlin format
    weight_perm = get_weight_perm(quant_type.size_bits)
    marlin_q_w_1 = marlin_weights(q_w, size_k, size_n, quant_type.size_bits,
                                  weight_perm)

    # Run Marlin repack GPU kernel
    marlin_q_w_2 = lightop.gemm_ops.awq_marlin_repack(
        q_w_awq,
        size_k,
        size_n,
        quant_type.size_bits,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(marlin_q_w_1, marlin_q_w_2)

####  pytest -v test_marlin_utils.py