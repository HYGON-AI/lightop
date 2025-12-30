# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年11月05日
description：测试MIOpen Adamw算子的性能，并与torch进行精度比较。
"""

import json

import pandas as pd
import torch
from torch.optim import adamw
from torch.profiler import profile, ProfilerActivity

from lightop import miopen_adamw


def compare_adamw_implementations(input_data):
    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # 准备测试数据
    input_tensor = [torch.randn(input_data, device='cuda', dtype=torch.float32)]
    grads = [torch.randn(input_data, device='cuda', dtype=torch.float32)]

    # 保存原始数据用于两个实现的独立测试
    original_input_tensor = input_tensor[0].clone().detach()
    original_grads = grads[0].clone().detach()

    print("=== 初始化参数 ===")
    print(f"参数张量形状: {input_tensor[0].shape}")
    print(f"参数范围: [{input_tensor[0].min():.6f}, {input_tensor[0].max():.6f}]")
    print(f"梯度范围: [{grads[0].min():.6f}, {grads[0].max():.6f}]")

    # ==================== 测试PyTorch AdamW ====================
    print("\n=== 测试PyTorch AdamW ===")

    # 为PyTorch实现创建独立的数据副本
    torch_input_tensor = [original_input_tensor.clone().detach()]
    torch_grads = [original_grads.clone().detach()]
    torch_exp_avgs = [torch.zeros_like(torch_input_tensor[0])]
    torch_exp_avg_sqs = [torch.zeros_like(torch_input_tensor[0])]
    torch_max_exp_avg_sqs = [torch.zeros_like(torch_input_tensor[0])]
    torch_state_steps = [torch.tensor(1, dtype=torch.float32)]

    amsgrad = False
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.001
    weight_decay = 0.001
    eps = 1e-8
    maximize = False

    # 执行PyTorch AdamW
    for i in range(10):
        torch_optimizer = adamw.adamw(
            params=torch_input_tensor,
            grads=torch_grads,
            exp_avgs=torch_exp_avgs,
            exp_avg_sqs=torch_exp_avg_sqs,
            max_exp_avg_sqs=torch_max_exp_avg_sqs,
            state_steps=torch_state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
        )

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        torch_optimizer = adamw.adamw(
            params=torch_input_tensor,
            grads=torch_grads,
            exp_avgs=torch_exp_avgs,
            exp_avg_sqs=torch_exp_avg_sqs,
            max_exp_avg_sqs=torch_max_exp_avg_sqs,
            state_steps=torch_state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
        )
    print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cuda_time_total'))

    torch_output = torch_input_tensor[0].clone().detach()
    torch_exp_avgs_output = torch_exp_avgs[0].clone().detach()
    torch_exp_avg_sqs_output = torch_exp_avg_sqs[0].clone().detach()
    torch_state_step = torch_state_steps[0].item()

    print(f"PyTorch AdamW执行完成")
    print(f"torch_state_steps: {torch_state_step}")

    # ==================== 测试MIOpen AdamW ====================
    print("\n=== 测试MIOpen AdamW ===")

    # 为MIOpen实现创建独立的数据副本
    miopen_input_tensor = [original_input_tensor.clone().detach()]
    miopen_grads = [original_grads.clone().detach()]
    miopen_exp_avgs = [torch.zeros_like(miopen_input_tensor[0])]
    miopen_exp_avg_sqs = [torch.zeros_like(miopen_input_tensor[0])]
    miopen_max_exp_avg_sqs = [torch.zeros_like(miopen_input_tensor[0])]
    miopen_state_steps = [torch.tensor(1, dtype=torch.float32)]

    # 执行MIOpen AdamW
    for i in range(10):
        miopen_optimizer = miopen_adamw(
            params=miopen_input_tensor,
            grads=miopen_grads,
            exp_avgs=miopen_exp_avgs,
            exp_avg_sqs=miopen_exp_avg_sqs,
            max_exp_avg_sqs=miopen_max_exp_avg_sqs,
            state_steps=miopen_state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
        )
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        miopen_optimizer = miopen_adamw(
            params=miopen_input_tensor,
            grads=miopen_grads,
            exp_avgs=miopen_exp_avgs,
            exp_avg_sqs=miopen_exp_avg_sqs,
            max_exp_avg_sqs=miopen_max_exp_avg_sqs,
            state_steps=miopen_state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
        )
    print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cuda_time_total'))

    miopen_output = miopen_input_tensor[0].clone().detach()
    miopen_exp_avgs_output = miopen_exp_avgs[0].clone().detach()
    miopen_exp_avg_sqs_output = miopen_exp_avg_sqs[0].clone().detach()
    miopen_state_step = miopen_state_steps[0].item()

    print(f"MIOpen AdamW执行完成")
    print(f"miopen_state_steps: {miopen_state_step}")

    # ==================== 精度对比 ====================
    print("\n=== 精度对比 ===")

    # 验证输出精度
    output_verify_result = torch.allclose(torch_output, miopen_output, rtol=1e-5, atol=1e-5, equal_nan=False)
    if output_verify_result:
        print("output精度校验: Passed")
    else:
        print("output精度校验: Failed")

    # 验证一阶动量
    avg_verify_result = torch.allclose(torch_exp_avgs_output, miopen_exp_avgs_output, rtol=1e-5, atol=1e-5,
                                       equal_nan=False)
    if avg_verify_result:
        print("exp_avg精度校验: Passed")
    else:
        print("exp_avg精度校验: Failed")

    # 验证二阶动量
    avg_sq_verify_result = torch.allclose(torch_exp_avg_sqs_output, miopen_exp_avg_sqs_output, rtol=1e-5, atol=1e-5,
                                          equal_nan=False)
    if avg_sq_verify_result:
        print("exp_avg_sq精度校验: Passed")
    else:
        print("exp_avg_sq精度校验: Failed")

    assert torch_state_steps == miopen_state_steps, "torch与MIOpen的迭代步长不一致"


if __name__ == "__main__":
    with open(r'./Adamw_test_size.txt', 'r', encoding='utf-8') as f:
        contents = f.readlines()

    test_size_df = pd.read_csv(r'../test_size/adamW.csv', encoding='utf-8')
    test_size_list = test_size_df['Size'].tolist()
    test_size_list_ = [json.loads(each.strip('\n')) for each in test_size_list]

    for idx, each_input in enumerate(test_size_list_):
        print("=" * 100)
        print(f"测试Size id: {idx}")
        compare_adamw_implementations(each_input)