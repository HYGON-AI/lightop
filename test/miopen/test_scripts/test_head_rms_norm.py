# -*- coding: UTF-8 -*-
"""
@Date    :2025/9/28 17:15  
@Author  :ZhangShuai
@Desc    :测试miopen_head_rms_norm函数
"""
import torch
import time
import lightop


def get_head_rms_norm_data(num_tokens, norm_head_num, head_dim, sp_size):
    """生成miopen_head_rms_norm测试数据"""
    # 确保参数为整数类型
    num_tokens = num_tokens
    norm_head_num = norm_head_num
    head_dim = head_dim
    sp_size = sp_size
    # 生成输入数据
    x = torch.randn(num_tokens, norm_head_num, head_dim, sp_size, dtype=torch.bfloat16).cuda()
    weight = torch.randn(norm_head_num * head_dim, dtype=torch.bfloat16).cuda()
    return x, weight


def test_head_rms_norm(num_tokens, norm_head_num, head_dim, sp_size, epsilon=1e-5):
    """测试miopen_head_rms_norm函数"""
    # 获取测试数据
    x, weight = get_head_rms_norm_data(num_tokens, norm_head_num, head_dim, sp_size)
    # 预热
    y = lightop.miopen_head_rms_norm(x, weight, epsilon, 0, norm_head_num, norm_head_num)
    torch.cuda.synchronize()

    # 性能测试
    start_time = time.time()
    for i in range(10):
        y = lightop.miopen_head_rms_norm(x, weight, epsilon, 0, norm_head_num, norm_head_num)
    torch.cuda.synchronize()
    end_time = time.time()

    # 计算平均时间
    avg_time = (end_time - start_time) * 1000 / 10
    print(f'miopen_head_rms_norm time: {avg_time:.3f} ms')
    # 输出信息
    print(f"Input x shape: {x.shape}")
    print(f"Input weight shape: {weight.shape}")
    print(f"Output y shape: {y.shape}")
    print(f"Parameters: num_tokens={num_tokens}, norm_head_num={norm_head_num}, head_dim={head_dim}, sp_size={sp_size}")
    print("-" * 50)

    return y


def read_rms_size(csv_path):
    """从CSV文件读取RMS参数"""
    import csv
    params_list = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            params = {
                'num_tokens': int(row['num_tokens']),
                'norm_head_num': int(row['norm_head_num']),
                'head_dim': int(row['head_dim']),
                'sp_size': int(row['sp_size'])
            }
            params_list.append(params)

    return params_list


if __name__ == '__main__':
    print('=' * 30 + '测试miopen_head_rms_norm多参数调用' + '=' * 30)
    try:
        rms_size_path = '../test_size/rms.csv'
        rms_params_list = read_rms_size(rms_size_path)

        for i, params in enumerate(rms_params_list):
            print(f'测试sizeid: {i + 1}')
            test_head_rms_norm(**params)

    except Exception as e:
        print(f"读取CSV文件失败: {e}")