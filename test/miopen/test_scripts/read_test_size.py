# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年08月02日
description: 支持测试脚本执行的辅助功能，读取各个算子的测试size
"""

import pandas as pd


def read_convfusion_size(size_file_path):
    convfusion_df = pd.read_csv(size_file_path)
    return [
        {
            'batch_size': row['n'],
            'channels': row['c'],
            'Height': row['H'],
            'Width': row['W'],
            'out_channels': row['k'],
            'kernel_h': row['y'],
            'kernel_w': row['x'],
            'stride_h': row['u'],
            'stride_w': row['v'],
            'pad_h': row['p'],
            'pad_w': row['q'],
            'dilation_h': row['l'],
            'dilation_w': row['j']
        }
        for _, row in convfusion_df.iterrows()
    ]


def read_gn_size(size_file_path):
    gn_df = pd.read_csv(size_file_path)
    return [{
        'batch_size': row['n'],
        'seqs': row['c'],
        'cols': row['H'],
        'rows': row['W'],
        'num_groups': row['g']
    }
        for _, row in gn_df.iterrows()]


def read_mha_size(size_file_path):
    gn_df = pd.read_csv(size_file_path)
    return [{
        'batch_size': row['B'],
        'head_nums': row['H'],
        'sequence_length': row['S'],
        'dim': row['D']
    }
        for _, row in gn_df.iterrows()]


def read_ln_size(size_file_path):
    ln_df = pd.read_csv(size_file_path)
    return [{
        'batch_size': row['dim1'],
        'seqs': row['dim2'],
        'cols': row['dim3']
    }
        for _, row in ln_df.iterrows()]


def read_ln5d_size(size_file_path):
    ln_df = pd.read_csv(size_file_path)
    return [{
        'batch_size': row['dim1'],
        'dima': row['dim2'],
        'dimb': row['dim3'],
        'seqs': row['dim4'],
        'cols': row['dim5']
    }
        for _, row in ln_df.iterrows()]


def read_moe_size(size_file_path):
    # 指定每列的数据类型（根据你的实际需求调整）
    dtype_dict = {
        'hidden_size': int,
        'num_tokens': int,
        'world_size': int,
        'num_shared_experts': int,
        'dp_size': int,
        'sp_size': int,
        'num_experts': int,
        'topk': int,
        'ep_size': int
    }
    ln_df = pd.read_csv(size_file_path, dtype=dtype_dict)
    return [{
        'hidden_size': row['hidden_size'],
        'num_tokens': row['num_tokens'],
        'world_size': row['world_size'],
        'num_shared_experts': row['num_shared_experts'],
        'dp_size': row['dp_size'],
        'sp_size': row['sp_size'],
        'num_experts': row['num_experts'],
        'topk': row['topk'],
        'ep_size': row['ep_size']
    }
        for _, row in ln_df.iterrows()]

def read_tt_size(size_file_path):
    ln_df = pd.read_csv(size_file_path)
    return [{
        'batch_size': row['n'],
        'seqs': row['c'],
        'cols': row['H'],
        'rows': row['W'],
        'transpose_mode': row['transpose_mode']
    }
        for _, row in ln_df.iterrows()]