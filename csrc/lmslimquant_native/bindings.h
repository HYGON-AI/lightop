// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>
#include <string>

namespace lightop::lmslimquant_native {

torch::Tensor rocblas_gemm_int8(
    torch::Tensor input,
    torch::Tensor weight,
    bool layout_tn);

torch::Tensor rocblas_gemm_fp16(
    torch::Tensor input,
    torch::Tensor weight,
    bool layout_tn);

torch::Tensor gemm_w4a16_v2(
    torch::Tensor input, torch::Tensor weight, torch::Tensor zeros_and_scales,
    int m, int n, int k, int stride_a, int stride_b, int stride_b_padded,
    int stride_c, int group_size, torch::Tensor workspace,
    int workspace_size, std::string config);

void awq_input_padding(torch::Tensor input, int m, int k, int stride_b_padded,
                       torch::Tensor workspace, int workspace_size);
void convert_s4_k_m8(torch::Tensor out_qweight, torch::Tensor out_scale_zero,
                     torch::Tensor workspace, torch::Tensor qweight,
                     torch::Tensor scales, torch::Tensor qzeros,
                     int m, int k, int group_size);
void permute_trans(torch::Tensor output, torch::Tensor input,
                   int64_t dim0, int64_t n);
void awq_dequant_colmajor(torch::Tensor output, torch::Tensor qweight,
                          torch::Tensor zeros_and_scales, int64_t k,
                          int64_t k_padded, int64_t n, int64_t group_size,
                          int64_t workspace_size);
void dequant_w4_gemm_colmajor(torch::Tensor output, torch::Tensor qweight,
                              torch::Tensor zeros_and_scales, int64_t k,
                              int64_t n, int64_t group_size);
torch::Tensor gptq_gemm_native(
    torch::Tensor input, torch::Tensor qweight, torch::Tensor qzeros,
    torch::Tensor scales, torch::Tensor g_idx, bool use_exllama, int64_t bit);
void gptq_shuffle_native(torch::Tensor qweight, torch::Tensor q_perm,
                         int64_t bit);

}  // namespace lightop::lmslimquant_native
