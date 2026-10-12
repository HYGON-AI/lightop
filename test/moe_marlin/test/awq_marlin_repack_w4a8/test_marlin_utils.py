# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional, Tuple
import pandas as pd
from marlin_quant import  unpack_int8_to_int4, w4a8_2_marlin_weight 

import lightop
import lightop.gemm_ops

@pytest.mark.parametrize("size_batch", [256])  # 可扩展其他batch值（如128）
@pytest.mark.parametrize("size_k, size_n", [
    (7168, 4096),   # TP1
    (2048, 7168),   # TP1
    (7168, 2048),   # TP2
    (1024, 7168),   # TP2
    (7168, 1024),   # TP4
    (512, 7168),    # TP4
    (7168, 512),    # TP8
    (256, 7168),    # TP8
])
def test_awq_marlin_repack_w4a8(size_batch, size_k, size_n):
    # （可选）提前验证参数合法性，与C++算子的TORCH_CHECK对齐
    assert size_k % 32 == 0, f"size_k={size_k} 必须是32的倍数（算子要求）"
    assert size_n % 64 == 0, f"size_n={size_n} 必须是64的倍数（算子要求）"
    
    # 2. 构造输入张量（直接使用参数化的size_k、size_n）
    native_weight = torch.randint(
        low=-127, 
        high=127, 
        size=(size_batch, size_n, size_k // 2),  # 维度匹配算子输入要求
        device="cuda:0", 
        dtype=torch.int8
    )
    
    # 3. 构造输出张量（预分配内存，维度由size_k、size_n计算）
    cuda_output = torch.zeros(
        size=(size_batch, size_k // 32, size_n * 32 // 8),  # 输出维度逻辑不变
        device="cuda:0", 
        dtype=torch.int32
    )
    
    # 4. 调用CUDA算子（直接传参数化的size_batch、size_k、size_n）
    try:
        lightop.gemm_ops.awq_marlin_repack_w4a8(
            native_weight, 
            cuda_output,
            size_batch, 
            size_k,
            size_n
        )
        torch.cuda.synchronize()  # 确保CUDA算子执行完成
    except Exception as e:
        pytest.fail(f"CUDA算子调用失败：{str(e)}")
    
    # 5. Python侧计算结果（用于对比验证）
    w_marlin_list = []
    for i in range(native_weight.shape[0]):
        w_marlin_in = w4a8_2_marlin_weight(native_weight[i])
        w_marlin_list.append(w_marlin_in)
    python_output = torch.stack(w_marlin_list, dim=0)
    
    # 6. 断言CUDA输出与Python输出一致（验证算子正确性）
    torch.testing.assert_close(
        python_output, 
        cuda_output,
        atol=0,  # 整数对比，绝对误差为0
        rtol=0,
        msg="CUDA算子输出与Python侧结果不一致"
    )

####  pytest -v test_marlin_utils.py

if __name__ == "__main__":

    # size_batch = 256
    # size_k = 7168
    # size_n = 512
    size_batch = 256
    size_k = 256
    size_n = 7168

    native_weight = torch.randint(-127, 127, (size_batch, size_n, size_k // 2), device="cuda:0", dtype=torch.int8)
    
    cuda_output = torch.zeros((size_batch, size_k // 32, size_n * 32 // 8), device="cuda:0", dtype=torch.int32)
    # ======================== cuda 算子权重处理 ===========================
    lightop.gemm_ops.awq_marlin_repack_w4a8(
                                            native_weight, 
                                            cuda_output,
                                            size_batch, 
                                            size_k,
                                            size_n
                                            )
    torch.cuda.synchronize()
    # ======================== python marlin 权重处理 =============================
    w_marlin_list = []
    for i in range(native_weight.shape[0]):
        w_marlin_in = w4a8_2_marlin_weight(native_weight[i])
        w_marlin_list.append(w_marlin_in)
    python_output = torch.stack(w_marlin_list, dim=0)

    torch.testing.assert_close(python_output, cuda_output)
    print("GEMM1 正确性验证通过！！！")

