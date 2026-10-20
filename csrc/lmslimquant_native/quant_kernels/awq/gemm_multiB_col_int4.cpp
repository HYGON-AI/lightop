// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "gemm_multiB_col_int4.h"
#include "basic_modules/gemm_weight_only_template.h"

int get_weight_only_workspace_size(int N_max, int K_max, int GroupSize) {
    using GemmWeightOnlyTool = gemm_weight_only_all::GemmWeightOnly;

    auto weight_only_gemm_workspace_size = GemmWeightOnlyTool::get_weight_only_workspace_size(N_max, K_max, GroupSize);

    return weight_only_gemm_workspace_size;
}

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
                          hipStream_t stream_id,
                          bool splitK_atomicAdd_available) // atomicAdd更快但可能有精度问题
{
    // 检查layout
    if(StrideA != StrideB) {
        printf("warning! StrideA=%d != StrideB=%d, input layout must be matA:RowMajor and matB:ColMajor!\n", StrideA, StrideB);
    }

    auto gemm_weight_only = gemm_weight_only_all::GemmWeightOnly{};
    gemm_weight_only.run_weight_only_gemm(A,
                                          B0,
                                          B1,
                                          C,
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideB_padded,
                                          StrideC,
                                          GroupSize,
                                          splitK_padA_workspace,      // 用于SplitK和tensorA添加pad的显存空间
                                          splitK_padA_workspace_size, // workspace有多少个字节
                                          stream_id,
                                          splitK_atomicAdd_available);
}
