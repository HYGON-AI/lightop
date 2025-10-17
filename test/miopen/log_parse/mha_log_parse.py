# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年09月29日
description: 解析MHA的测试日志
"""

import re
from collections import defaultdict

import pandas as pd

from utils import generate_parsed_file_path, split_lines


def extract_groupnorm_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    contents = content.split('=' * 50)
    contents.pop()

    test_res = defaultdict(list)
    pattern_for_size = r'size大小\s*batch_size:\s*(\d+),\s*head_nums:\s*(\d+),\s*sequence_length:\s*(\d+),\s*dim:\s*(\d+)'  # 匹配batch,heads,sequence,dim参数
    pattern_for_test_type = r'test_type:\s+(\w+)'  # 匹配测试类型
    pattern_for_result = r'verify:\s+(.*)'
    pattern_for_time = r'torch_time: (\d+\.\d+) ms\. mlop_time: (\d+\.\d+) ms\. speedup: (-?\d+\.\d+)'
    for each_content in contents:
        size_match = re.search(pattern_for_size, each_content)
        if size_match:
            batches, heads, seqs, dim = map(int, size_match.groups())
            test_res['B'].append(batches)
            test_res['H'].append(heads)
            test_res['S'].append(seqs)
            test_res['D'].append(dim)
        try:
            test_type = re.search(pattern_for_test_type, each_content).group(1)
        except AttributeError:
            print('报错content:', each_content)
        time_match = re.search(pattern_for_time, each_content)
        if time_match:
            torch_time, mlop_time, speedup = map(float, time_match.groups())
            test_res[f'{test_type}_torch_time(ms)'].append(torch_time)
            test_res[f'{test_type}_mlop_time(ms)'].append(mlop_time)
            test_res[f'{test_type}_speedup'].append(speedup)
            verify_result = re.search(pattern_for_result, each_content).group(1)
            test_res[f'{test_type}_verify'].append(verify_result)
        else:
            raise ValueError('未提取到用例执行耗时，请检查正则表达式')

    # 因为size [B,H,S,D]在行维度上进行追加，耗时在列维度上进行追加，会产生size行重复的现象，所以对size行进行切片
    test_res = split_lines(test_res, filter_cols=['B', 'H', 'S', 'D'])
    test_res_df = pd.DataFrame(test_res)
    parsed_log_path = generate_parsed_file_path(log_path)
    test_res_df.to_csv(parsed_log_path, index=False, encoding='utf-8')
    print(f'日志解析完成，结果保存路径：{parsed_log_path}')


if __name__ == '__main__':
    # 将测试日志路径修改为真实文件路径
    log_path = r'your_log_path'
    extract_groupnorm_log(log_path)
