# -*- coding: UTF-8 -*-
"""
@Date    :2025/9/28 17:15  
@Author  :ZhangShuai
@Desc    :测试miopen_rms_norm_dynamic_quant函数的多参数调用
"""
import torch
import time
import pandas as pd
import lightop


def get_rms_norm_data(num_tokens, hidden_size):
    """生成RMS Norm测试数据"""
    # 确保参数为整数类型
    num_tokens = int(num_tokens)
    hidden_size = int(hidden_size)
    # 生成输入数据
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16).cuda()
    x2 = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16).cuda()
    weight = torch.randn(hidden_size, dtype=torch.bfloat16).cuda()
    smoothScale = torch.randn(hidden_size, dtype=torch.bfloat16).cuda()
    return x, x2, weight, smoothScale


def test_add_rms_norm_dynamic_quant(num_tokens, hidden_size, epsilon=1e-5):
    """测试带残差的miopen_add_rms_norm_dynamic_quant函数"""
    # 获取测试数据
    x, x2, weight, smoothScale = get_rms_norm_data(num_tokens, hidden_size)
    # 预热
    after_res, y, per_token_scale = lightop.miopen_add_rms_norm_dynamic_quant(
        x, x2, weight, smoothScale, epsilon
    )
    torch.cuda.synchronize()

    # 性能测试
    start_time = time.time()
    for i in range(10):
        after_res, y, per_token_scale = lightop.miopen_add_rms_norm_dynamic_quant(
            x, x2, weight, smoothScale, epsilon
        )
    torch.cuda.synchronize()
    end_time = time.time()

    # 计算平均时间
    avg_time = (end_time - start_time) * 1000 / 10
    print(f'miopen_add_rms_norm_dynamic_quant time: {avg_time:.3f} ms')

    # 输出信息
    print(f"Input x shape: {x.shape}")
    print(f"Input x2 shape: {x2.shape}")
    print(f"Output after_res shape: {after_res.shape}")
    print(f"Output y shape: {y.shape}")
    print(f"Output per_token_scale shape: {per_token_scale.shape}")
    print(f"Parameters: num_tokens={num_tokens}, hidden_size={hidden_size}")
    print("-" * 50)

    return after_res, y, per_token_scale


def test_rms_norm_dynamic_quant_without_residual(num_tokens, hidden_size, epsilon=1e-5):
    """测试不带残差的miopen_rms_norm_dynamic_quant_without_residual函数"""
    # 获取测试数据
    x, _, weight, smoothScale = get_rms_norm_data(num_tokens, hidden_size)
    # 预热
    after_res, y, per_token_scale = lightop.miopen_rms_norm_dynamic_quant_without_residual(
        x, weight, smoothScale, epsilon
    )
    torch.cuda.synchronize()
    # 性能测试
    start_time = time.time()
    for i in range(10):
        after_res, y, per_token_scale = lightop.miopen_rms_norm_dynamic_quant_without_residual(
            x, weight, smoothScale, epsilon
        )
    torch.cuda.synchronize()
    end_time = time.time()
    # 计算平均时间
    avg_time = (end_time - start_time) * 1000 / 10
    print(f'miopen_rms_norm_dynamic_quant_without_residual time: {avg_time:.3f} ms')

    # 输出信息
    print(f"Input x shape: {x.shape}")
    print(f"Input weight shape: {weight.shape}")
    print(f"Output y shape: {y.shape}")
    print(f"Output per_token_scale shape: {per_token_scale.shape}")
    print(f"Parameters: num_tokens={num_tokens}, hidden_size={hidden_size}")
    print("-" * 50)

    return after_res, y, per_token_scale


def read_rms_norm_size(size_file_path):
    """从CSV文件读取RMS Norm参数"""
    dtype_dict = {
        'num_tokens': int,
        'total_head_num': int,
        'norm_head_start': int,
        'norm_head_num': int,
        'head_dim': int,
        'hidden_size': int,
        'sp_size': int
    }
    ln_df = pd.read_csv(size_file_path, dtype=dtype_dict)
    return [{
        'num_tokens': row['num_tokens'],
        'hidden_size': row['hidden_size']
    } for _, row in ln_df.iterrows()]


if __name__ == '__main__':
    print('=' * 30 + '测试RMS Norm动态量化多参数调用' + '=' * 30)
    rms_size_path = '../test_size/rms.csv'
    rms_params_list = read_rms_norm_size(rms_size_path)
    # 测试带残差的版本
    print("测试带残差的 RMS Norm 动态量化:")
    for i, params in enumerate(rms_params_list):
        print(f'测试sizeid: {i + 1}')
        test_add_rms_norm_dynamic_quant(**params)
    print("\n" + "=" * 80 + "\n")
    # 测试不带残差的版本
    print("测试不带残差的 RMS Norm 动态量化:")
    for i, params in enumerate(rms_params_list):
        print(f'测试sizeid: {i + 1}')
        test_rms_norm_dynamic_quant_without_residual(**params)