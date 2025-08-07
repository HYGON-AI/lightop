# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年08月01日
description: 支持日志解析的公共函数
"""

import os


def generate_parsed_file_path(log_path: str) -> str:
    """根据原始日志文件路径生成解析后的日志文件路径。"""
    dir_name = os.path.dirname(log_path)  # 获取文件路径：'/home/user/documents/'
    base_name = os.path.basename(log_path)  # 获取文件名：'file.txt'
    file_name, _ = os.path.splitext(base_name)  # 获取文件名：'file' 以及后缀 '.txt'
    parsed_log_path = os.path.join(dir_name, file_name + '_parsed' + '.csv')
    return parsed_log_path


def process_time(time_word: str) -> float:
    """将不同的时间单位如us, s等统一为ms。"""
    if time_word.endswith('us'):
        time_word = time_word.strip('us')
        time_count = round(float(time_word) / 1000, 3)
    elif time_word.endswith('ms'):
        time_word = time_word.strip('ms')
        time_count = float(time_word)
    elif time_word.endswith('s'):
        time_word = time_word.strip('s')
        time_count = round(float(time_word) * 1000, 3)
    else:
        raise ValueError(f'无效的时间单位: {time_word}')
    return time_count

