// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

/*
    根据模型的size，获取需要的显存空间
*/
int get_weight_only_workspace_size(int N_max,             // 模型权重的size N
                                   int K_max,             // 模型权重sizeK最大值，计算SplitK所需显存空间
                                   int GroupSize);        // 计算添加padding的大小为 2 * GroupSize

/*
执行gemm计算
输入的特征矩阵，size=(M, K)，行主序，stride=(K, 1)
输入的权重矩阵，压缩前，size= (K_Padded, N)，行主序，stride=(1, K_Padded)
输出矩阵，size=(M, N)，行主序，stride=(N, 1)
*/
void run_weight_only_gemm(const void* A,
                          const void* B0,
                          const void* B1,
                          void* C,
                          int M,
                          int N,
                          int K,
                          int StrideA,
                          int StrideB,
                          int StrideB_padded, // 输入的权重矩阵添加pad后的K
                          int StrideC,
                          int GroupSize,
                          void* splitK_padA_workspace,    // 用于SplitK和tensorA添加pad的显存空间
                          int splitK_padA_workspace_size, // workspace有多少个字节
                          hipStream_t stream_id           = 0,
                          bool splitK_atomicAdd_available = false); // atomicAdd更快但可能有精度问题
