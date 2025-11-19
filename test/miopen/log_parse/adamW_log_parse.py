# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年11月19日
description: 解析AdamW的测试日志，提取Lightop AdamW以及Torch AdamW的接口耗时，并验证接口的输出精度是否对齐
"""

import re
import pandas as pd


def parse_log_file(file_path):
    # 读取日志文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分割成不同的测试用例
    test_cases = content.split('测试Size id:')

    results = []

    for case in test_cases[1:]:  # 跳过第一个空元素
        # 提取size_id
        size_id_match = re.search(r'(\d+)', case)
        if not size_id_match:
            continue

        size_id = size_id_match.group(1)

        # 提取PyTorch AdamW的CUDA时间
        pytorch_cuda_time_match = re.search(r'Self CUDA time total:\s*([\d.]+)us', case)
        pytorch_cuda_time = pytorch_cuda_time_match.group(1) if pytorch_cuda_time_match else "N/A"

        # 提取MIOpen AdamW的CUDA时间
        miopen_cuda_time_match = re.search(r'Self CUDA time total:\s*([\d.]+)us',
                                           case[pytorch_cuda_time_match.end():] if pytorch_cuda_time_match else case)
        miopen_cuda_time = miopen_cuda_time_match.group(1) if miopen_cuda_time_match else "N/A"

        # 提取精度校验结果
        output_check = "Passed" if "output精度校验: Passed" in case else "Failed"
        exp_avg_check = "Passed" if "exp_avg精度校验: Passed" in case else "Failed"
        exp_avg_sq_check = "Passed" if "exp_avg_sq精度校验: Passed" in case else "Failed"

        results.append({
            'size_id': size_id,
            'pytorch_cuda_time': pytorch_cuda_time,
            'miopen_cuda_time': miopen_cuda_time,
            'output_check': output_check,
            'exp_avg_check': exp_avg_check,
            'exp_avg_sq_check': exp_avg_sq_check
        })

    return results


def display_results_table(results):
    # 打印表头
    print(
        f"{'Size ID':<8} {'PyTorch CUDA Time (us)':<20} {'MIOpen CUDA Time (us)':<20} {'Output':<10} {'Exp_avg':<10} {'Exp_avg_sq':<10}")
    print("-" * 85)

    # 打印每一行数据
    for result in results:
        print(f"{result['size_id']:<8} {result['pytorch_cuda_time']:<20} {result['miopen_cuda_time']:<20} "
              f"{result['output_check']:<10} {result['exp_avg_check']:<10} {result['exp_avg_sq_check']:<10}")


# 使用示例
if __name__ == "__main__":
    # 解析日志文件，日志路径和解析后的日志路径，根据实际情况进行修改
    results = parse_log_file('./log_adamw_1107.txt')
    df = pd.DataFrame(results)
    df.to_csv('./log_test_result.csv', index=False)

    # 显示表格
    display_results_table(results)

    # 统计通过率
    total_tests = len(results)
    output_passed = sum(1 for r in results if r['output_check'] == 'Passed')
    exp_avg_passed = sum(1 for r in results if r['exp_avg_check'] == 'Passed')
    exp_avg_sq_passed = sum(1 for r in results if r['exp_avg_sq_check'] == 'Passed')

    print(f"\n统计信息:")
    print(f"总测试数: {total_tests}")
    print(f"Output精度通过率: {output_passed}/{total_tests} ({output_passed / total_tests * 100:.1f}%)")
    print(f"Exp_avg精度通过率: {exp_avg_passed}/{total_tests} ({exp_avg_passed / total_tests * 100:.1f}%)")
    print(f"Exp_avg_sq精度通过率: {exp_avg_sq_passed}/{total_tests} ({exp_avg_sq_passed / total_tests * 100:.1f}%)")