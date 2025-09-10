# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年08月01日
description: 解析groupnorm及groupnorm_silu日志，解析groupnorm_silu日志时需打开weather_gn_silu开关
"""

import re
from collections import defaultdict

import pandas as pd

from utils import generate_parsed_file_path, split_lines


def extract_groupnorm_log(log_path, gn_silu=False):
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    contents = content.split('size_id')
    contents.pop(0)

    test_res = defaultdict(list)
    pattern_for_size = r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*.*\)'  # 匹配n,c,h,w参数
    pattern_for_test_type = r'\(.*?(\b\w+\b)\s*,\s*(\b\w+\b)\)'  # 匹配精度类型及布局
    pattern_for_result = r'verify:\s+(.*)'
    if gn_silu:
        pattern_for_time = r'miopen_gn_torch_silu time: (\d+\.\d+) ms\. miopen time: (\d+\.\d+) ms speedup: (-?\d+\.\d+)'
    else:
        pattern_for_time = r'lightop_time: (\d+\.\d+) ms\. miopen_lightop_time: (\d+\.\d+) ms\. speedup: (-?\d+\.\d+)'
    for each_content in contents:
        size_match = re.search(pattern_for_size, each_content)
        if size_match:
            n, c, h, w = map(int, size_match.groups())
            test_res['n'].append(n)
            test_res['c'].append(c)
            test_res['h'].append(h)
            test_res['w'].append(w)
        test_type = '_'.join(re.findall(pattern_for_test_type, each_content)[0])
        time_match = re.search(pattern_for_time, each_content)
        if time_match:
            lightop_time, miopen_lightop_time, speedup = map(float, time_match.groups())
            if gn_silu:
                test_res['miopen_gn_torch_silu_time(ms)'].append(lightop_time)
            else:
                test_res[f'{test_type}_lightop_time(ms)'].append(lightop_time)
            test_res[f'{test_type}_miopen_lightop_time(ms)'].append(miopen_lightop_time)
            test_res[f'{test_type}_speedup'].append(speedup)
            verify_result = re.search(pattern_for_result, each_content).group(1)
            test_res[f'{test_type}_verify_result'].append(verify_result)
        else:
            raise ValueError('未提取到用例执行耗时，请检查正则表达式')

    # 因为size [n,c,h,w]在行维度上进行追加，耗时在列维度上进行追加，会产生size行重复的现象，所以对size行进行切片
    test_res = split_lines(test_res)
    test_res_df = pd.DataFrame(test_res)
    parsed_log_path = generate_parsed_file_path(log_path)
    test_res_df.to_csv(parsed_log_path, index=False, encoding='utf-8')
    print(f'日志解析完成，结果保存路径：{parsed_log_path}')


if __name__ == '__main__':
    # 将测试日志路径修改为真实文件路径
    log_path = r'your_log_path'
    # 若是groupnorm算子测试，请将weather_gn_silu的值设置为False
    weather_gn_silu = False
    extract_groupnorm_log(log_path, weather_gn_silu)
