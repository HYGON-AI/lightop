# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年07月30日
description: 解析conv_bias以及conv_bias_add测试日志
"""

import os
import re
from collections import defaultdict

import pandas as pd

from utils import generate_parsed_file_path, process_time


def extract_conv_bias_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    contents = content.split('测试sizeid')
    contents.pop(0)

    test_res_save = defaultdict(list)
    pattern_torch_api_time = r'Self torch api time:\s+(\d+.\d+)'
    pattern_miopen_api_time = r'Self miopen api time:\s+(\d+.\d+)'
    pattern_cuda_time = r'Self CUDA time total:\s+(\d+.\d+(?:ms|us|s))'
    for each_content in contents:
        test_res_save['torch_api(ms)'].append(re.search(pattern_torch_api_time, each_content).group(1))
        torch_prof_time, miopen_prof_time = re.findall(pattern_cuda_time, each_content)
        torch_prof_time = process_time(torch_prof_time)
        miopen_prof_time = process_time(miopen_prof_time)
        test_res_save['torch_prof(ms)'].append(torch_prof_time)
        test_res_save['miopen_api(ms)'].append(re.search(pattern_miopen_api_time, each_content).group(1))
        test_res_save['miopen_prof(ms)'].append(miopen_prof_time)
    test_res_df = pd.DataFrame(test_res_save)
    parsed_log_path = generate_parsed_file_path(log_path)
    test_res_df.to_csv(parsed_log_path, index=False, encoding='utf-8')
    print(f'日志解析完成，结果保存路径：{parsed_log_path}')


if __name__ == '__main__':
    # 请将测试路径修改为真实文件路径
    log_file_path = 'your_log_path'
    extract_conv_bias_log(log_file_path)