// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "lmslimquant_native/quant_kernels/base/base.cuh"
//#include "lmslimquant_native/quant_kernels/awq/gemm_multiB_col_int4.h"
#include <torch/extension.h>
#include "lmslimquant_native/quant_kernels/awq/kernels/format.cuh"
#include "lmslimquant_native/quant_kernels/smoothquant/smquant_gemm.cuh"
#include <torch/all.h>
// #include "lmslimquant_native/quant_kernels/awq/gemm_w4a16.h"
#include "lmslimquant_native/quant_kernels/awq/gemm_w4a16_v2.h"


torch::Tensor run_gemm_w4a16_v2(torch::Tensor input,
                    torch::Tensor weight,
                    torch::Tensor zeros_and_scales,
                    //torch::Tensor output,
                    int m,
                    int n,
                    int k,
                    int StrideA,
                    int StrideB,
                    int StrideB_padded,
                    int StrideC,
                    int groupsize,
                    torch::Tensor splitK_padA_workspace,
                    int splitK_padA_workspace_size,
                    string modelsize_key_ptr
                    );

void run_awq_input_padding(torch::Tensor input, int m,int k,int StrideB_padded, torch::Tensor splitK_padA_workspace,int splitK_padA_workspace_size);


void run_dequant_gemm_rocblas_gpu(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor deweight,
        torch::Tensor zeros_and_scales,
        torch::Tensor output,
        int m,
        int n,
        int k,
        int group_size);

void run_convert_s4_k_m8(torch::Tensor _qw,
                            torch::Tensor _sz,
                            torch::Tensor _ws,
                            torch::Tensor qw,
                            torch::Tensor scales,
                            torch::Tensor qzeros,
                            int   m,
                            int   k,
                            int   group_size);

void run_permute_trans(torch::Tensor _sz,
                        torch::Tensor sz,
                        int64_t dim0,
                        int64_t n
                        );
void run_awq_dequant_colmajor(torch::Tensor dqweight,
                                torch::Tensor qweight,
                                torch::Tensor zeros_and_scales,
                                int64_t k,
                                int64_t k_pad,
                                int64_t n,
                                int64_t group_size,
                                int64_t workspace_size);
void run_dequant_w4_gemm_colmajor(torch::Tensor dqweight,
                                torch::Tensor qweight,
                                torch::Tensor zeros_and_scales,
                                int64_t k,
                                int64_t n,
                                int64_t group_size);

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit);

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit);


torch::Tensor rocblas_gemm_int8(torch::Tensor input,torch::Tensor weight,bool layout_tn);
// torch::Tensor cutlass_gemm_w8a8(torch::Tensor input,
//                                     torch::Tensor weight,
//                                     torch::Tensor scale_x,
//                                     torch::Tensor scale_y,
//                                     torch::Tensor bias,
//                                     int m,
//                                     int n,
//                                     int k
//                                     );
torch::Tensor rocblas_gemm_fp16(torch::Tensor input,torch::Tensor weight,bool layout_tn);