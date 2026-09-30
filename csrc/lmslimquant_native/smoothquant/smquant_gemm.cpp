// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "lmslimquant_native/smoothquant/smquant_gemm.cuh"

void gemm_rocblas_int8_gpu(hipStream_t stream,
                            rocblas_handle handle,
                                    int32_t* output,
                                    const int8_t* input,
                                    const int8_t* weight,
                                    const int M,
                                    const int N,
                                    const int K,
                                    bool layout_tn)
{
    //参数说明
    // 矩阵乘法，将结果存储在矩阵C中
    const int32_t alpha = 1;
    const int32_t beta = 0;
    //weight shape是NN的shape，但是实际的layout是TN

    //需要注意处理非32的倍数时候的操作，需要进行pad
    //  tn
    if(layout_tn){
        rocblasGemmEx(handle,
                        rocblas_operation_transpose,
                        rocblas_operation_none,
                        N,
                        M,
                        K,
                        alpha,
                        weight,
                        K,       //lda
                        input,
                        K,       //ldb
                        beta,
                        output,
                        N        //ldc
                        );
    }
    //nn
    else {
        rocblasGemmEx(handle,
                        rocblas_operation_none,
                        rocblas_operation_none,
                        N,
                        M,
                        K,
                        alpha,
                        weight,
                        N,
                        input,
                        K,       //ldb
                        beta,
                        output,
                        N        //ldc
                        );
        }
    }
