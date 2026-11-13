// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <string>

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

#include <rocblas/rocblas.h>
#include <hiprand/hiprand.h>

/*
根据运行的batchsize和模型size，获取需要的显存空间
*/
int get_w4a16_workspace_size(int M_max, int N_max);

/*
执行gemm计算
    输入的特征矩阵，size=(M, K)，行主序，stride=(K, 1)
    输入的权重矩阵，压缩前，size= (K, N)，列主序，stride=(1, K)
    输出矩阵，size=(M, N)，行主序，stride=(N, 1)
*/
void gemm_w4a16_bw(const void* A,
                const void* B0,
                const void* B1,
                void* C,
                int M,
                int N,
                int K,
                int StrideA,
                int StrideB,
                int StrideC,
                int group_size,
                void* workspace,    // 用于计算的显存空间
                int workspace_size, // workspace有多少个字节
                hipStream_t stream_id);

void gemm_w4a16(const void* A,
                const void* B0,
                const void* B1,
                void* C,
                int M,
                int N,
                int K,
                int StrideA,
                int StrideB,
                int StrideC,
                int group_size,
                void* workspace,    // 用于计算的显存空间
                int workspace_size, // workspace有多少个字节
                hipStream_t stream_id);

/*
执行反量化计算，将列主序的weight矩阵进行反量化
    输入的权重矩阵，压缩前，size= (K, N)，列主序，stride=(1, K)
*/
int dequant_w4_gemm_colmajor(const void* B0,
                             const void* B1,
                             void* workspace,    // 用于计算的显存空间
                             int workspace_size, // workspace有多少个字节
                             int N,
                             int K,
                             int StrideB,
                             int group_size,
                             hipStream_t stream_id);

int dequant_w4_gemm_colmajor_trans(const void* B0,
                            const void* B1,
                            void* workspace,    // 用于计算的显存空间
                            int workspace_size, // workspace有多少个字节
                            int N,
                            int K,
                            int StrideB,
                            int group_size,
                            hipStream_t stream_id);

int input_padding(const void* input,
                  void* workspace,    // 用于计算的显存空间
                  int workspace_size, // workspace有多少个字节
                  const int M,
                  const int K,
                  const int K_Padded,
                  hipStream_t stream_id);
