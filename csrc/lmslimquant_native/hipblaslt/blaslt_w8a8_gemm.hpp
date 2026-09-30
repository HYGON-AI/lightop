// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_bf16.h>
#include <library_types.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include <torch/extension.h>

#include <mutex>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <sstream>

namespace hipblaslt_gemm {
hipblasOperation_t get_transpose(char trans){
    if(trans == 'N' || trans == 'n')
        return HIPBLAS_OP_N;
    else
        return HIPBLAS_OP_T;
}

struct BlasltGlobalContext {
    hipblasLtHandle_t handle;

    BlasltGlobalContext() {
        hipblasLtCreate(&handle);
    }

    ~BlasltGlobalContext() {
        hipblasLtDestroy(handle);
    }
};

#define MAX_GPU 16

class BlasltHandle {
public:
    static hipblasLtHandle_t get() {
        int dev;
        hipGetDevice(&dev);
        hipSetDevice(dev);
        std::call_once(init_flags_[dev], [&](){
             hipblasLtCreate(&handles_[dev]);
        });
        return handles_[dev];
    }

    static void destroy_all() {
        for (int i = 0; i < MAX_GPU; ++i) {
            if (handles_[i]) {
                hipblasLtDestroy(handles_[i]);
                handles_[i] = nullptr;
            }
        }
    }

private:
    static inline std::once_flag init_flags_[MAX_GPU];
    static inline hipblasLtHandle_t handles_[MAX_GPU] = {nullptr};
};

class BlasltMatmulDesc {
public:
    static hipblasLtMatmulDesc_t get() {
        int dev;
        hipGetDevice(&dev);
        hipSetDevice(dev);
        std::call_once(init_flags_[dev], [&](){
            hipblasLtMatmulDescCreate(&desc_[dev], HIPBLAS_COMPUTE_32I, HIP_R_32F);
            hipblasLtMatmulDescSetAttribute(desc_[dev],
                    HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, (void*)&mode_, sizeof(void*));
            hipblasLtMatmulDescSetAttribute(desc_[dev],
                    HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, (void*)&mode_, sizeof(void*));
            hipblasLtMatmulDescSetAttribute(desc_[dev],
                    HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a_, sizeof(int32_t));
            hipblasLtMatmulDescSetAttribute(desc_[dev],
                    HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b_, sizeof(int32_t));
        });
        return desc_[dev];
    }
private:
    static inline hipblasOperation_t trans_a_ = HIPBLAS_OP_T;
    static inline hipblasOperation_t trans_b_ = HIPBLAS_OP_N;
    static inline hipblasLtMatmulMatrixScale_t mode_ = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    static inline std::once_flag init_flags_[MAX_GPU];
    static inline hipblasLtMatmulDesc_t desc_[MAX_GPU] = {nullptr};
};

void w8a8_fp16_gemm(const torch::Tensor& mat_a, const torch::Tensor& mat_b,
        const torch::Tensor& scale_a, const torch::Tensor& scale_b,
        torch::Tensor& output,
        const int64_t m, const int64_t n, const int64_t k, const int64_t batch,
        const std::string& transpose_flag,
        const torch::Tensor& alpha, const torch::Tensor& beta,
        const std::optional<torch::Tensor>& bias) {

    hipblasLtHandle_t handle = BlasltHandle::get();

    hipblasOperation_t trans_a = get_transpose(transpose_flag[0]);
    hipblasOperation_t trans_b = get_transpose(transpose_flag[1]);

    int64_t stride_a = n * k;
    int64_t stride_b = m * k;
    int64_t stride_c = m * n;
    int64_t stride_output = m * n;

    size_t row_a = k;
    size_t col_a = n;
    size_t row_b = k;
    size_t col_b = m;
    size_t lda   = k;
    size_t ldb   = k;
    size_t ldc   = n;
    size_t ldd   = n;

    hipblasLtMatrixLayout_t mat_a_layout, mat_b_layout, mat_c_layout, output_layout;
    hipblasLtMatrixLayoutCreate(&mat_a_layout, HIP_R_8I, row_a, col_a, lda);
    hipblasLtMatrixLayoutCreate(&mat_b_layout, HIP_R_8I, row_b, col_b, ldb);
    hipblasLtMatrixLayoutCreate(&mat_c_layout, HIP_R_32I, n, m, ldc);
    hipblasLtMatrixLayoutCreate(&output_layout, HIP_R_16F, n, m, ldd);

    hipblasLtMatrixLayoutSetAttribute(mat_a_layout,
            HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch, sizeof(int32_t));
    hipblasLtMatrixLayoutSetAttribute(mat_a_layout,
            HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &(stride_a), sizeof(int64_t));

    hipblasLtMatrixLayoutSetAttribute(mat_b_layout,
            HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch, sizeof(int32_t));
    hipblasLtMatrixLayoutSetAttribute(mat_b_layout,
            HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &(stride_b), sizeof(int64_t));

    hipblasLtMatrixLayoutSetAttribute(mat_c_layout,
            HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch, sizeof(int32_t));
    hipblasLtMatrixLayoutSetAttribute(mat_c_layout,
            HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &(stride_c), sizeof(int64_t));

    hipblasLtMatrixLayoutSetAttribute(output_layout,
            HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch, sizeof(int32_t));
    hipblasLtMatrixLayoutSetAttribute(output_layout,
            HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &(stride_output), sizeof(int64_t));

    hipblasLtMatmulDesc_t matmul_desc = BlasltMatmulDesc::get();

    void* scale_a_ptr = reinterpret_cast<void*>(scale_a.data_ptr());
    void* scale_b_ptr = reinterpret_cast<void*>(scale_b.data_ptr());

    hipblasLtMatmulDescSetAttribute(matmul_desc,
            HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            reinterpret_cast<void*>(&scale_a_ptr),
            sizeof(void*));
    hipblasLtMatmulDescSetAttribute(matmul_desc,
            HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            reinterpret_cast<void*>(&scale_b_ptr),
            sizeof(void*));

    if (bias.has_value()) {
        hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
        void* bias_ptr = reinterpret_cast<void*>(bias.value().data_ptr());
        hipDataType bias_type = HIP_R_16F;
        hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(int32_t));
        hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(int32_t));
        hipblasLtMatmulDescSetAttribute(matmul_desc,
                HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                reinterpret_cast<void*>(&bias_ptr),
                sizeof(void*));
    }

    // 设置查询信息
    hipblasLtMatmulHeuristicResult_t heuristicResult[1];
    hipblasLtMatmulPreference_t pref;
    int returnedAlgoCount = 0;
    hipblasLtMatmulPreferenceCreate(&pref);
    int max_workspace_size=0;
    hipblasLtMatmulPreferenceSetAttribute(
            pref,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &max_workspace_size,
            sizeof(max_workspace_size));
    // 自动寻优配置
    auto ret = hipblasLtMatmulAlgoGetHeuristic(
            handle, matmul_desc,
            mat_a_layout, mat_b_layout, mat_c_layout, output_layout,
            pref, 1, heuristicResult, &returnedAlgoCount);
    if (ret != HIPBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
        std::cerr << "[BLASLT] No suitable hipBLASLt algo found, m: " << m << ", n: " << n << ", k: " << k << std::endl;
    }
    void* mat_c_fake_ptr = reinterpret_cast<void*>(0x01);
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    hipblasLtMatmul(
            handle, matmul_desc, alpha.data_ptr(),
            mat_a.data_ptr(), mat_a_layout,
            mat_b.data_ptr(), mat_b_layout,
            beta.data_ptr(),
            mat_c_fake_ptr, mat_c_layout,
            output.data_ptr(), output_layout,
            &heuristicResult[0].algo, nullptr, 0, stream);
    hipblasLtMatrixLayoutDestroy(mat_a_layout);
    hipblasLtMatrixLayoutDestroy(mat_b_layout);
    hipblasLtMatrixLayoutDestroy(mat_c_layout);
    hipblasLtMatrixLayoutDestroy(output_layout);
    hipblasLtMatmulPreferenceDestroy(pref);
    return ;
}

void w8a8_bf16_gemm(const torch::Tensor& mat_a, const torch::Tensor& mat_b,
        const torch::Tensor& scale_a, const torch::Tensor& scale_b,
        torch::Tensor& output,
        const int64_t m, const int64_t n, const int64_t k, const int64_t batch,
        const std::string& transpose_flag,
        const torch::Tensor& alpha, const torch::Tensor& beta,
        const std::optional<torch::Tensor>& bias) {
    hipblasLtHandle_t handle = BlasltHandle::get();
    hipblasOperation_t trans_a = get_transpose(transpose_flag[0]);
    hipblasOperation_t trans_b = get_transpose(transpose_flag[1]);

    int64_t stride_a = n * k;
    int64_t stride_b = m * k;
    int64_t stride_c = m * n;
    int64_t stride_output = m * n;

    size_t row_a = k;
    size_t col_a = n;
    size_t row_b = k;
    size_t col_b = m;
    size_t lda   = k;
    size_t ldb   = k;
    size_t ldc   = n;
    size_t ldd   = n;

    hipblasLtMatrixLayout_t mat_a_layout, mat_b_layout, mat_c_layout, output_layout;
    hipblasLtMatrixLayoutCreate(&mat_a_layout, HIP_R_8I, row_a, col_a, lda);
    hipblasLtMatrixLayoutCreate(&mat_b_layout, HIP_R_8I, row_b, col_b, ldb);
    hipblasLtMatrixLayoutCreate(&mat_c_layout, HIP_R_32I, n, m, ldc);
    hipblasLtMatrixLayoutCreate(&output_layout, HIP_R_16BF, n, m, ldd);

    hipblasLtMatrixLayoutSetAttribute(mat_a_layout,
            HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch, sizeof(int32_t));
    hipblasLtMatrixLayoutSetAttribute(mat_a_layout,
            HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &(stride_a), sizeof(int64_t));

    hipblasLtMatrixLayoutSetAttribute(mat_b_layout,
            HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch, sizeof(int32_t));
    hipblasLtMatrixLayoutSetAttribute(mat_b_layout,
            HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &(stride_b), sizeof(int64_t));

    hipblasLtMatrixLayoutSetAttribute(mat_c_layout,
            HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch, sizeof(int32_t));
    hipblasLtMatrixLayoutSetAttribute(mat_c_layout,
            HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &(stride_c), sizeof(int64_t));

    hipblasLtMatrixLayoutSetAttribute(output_layout,
            HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch, sizeof(int32_t));
    hipblasLtMatrixLayoutSetAttribute(output_layout,
            HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &(stride_output), sizeof(int64_t));

    hipblasLtMatmulDesc_t matmul_desc = BlasltMatmulDesc::get();

    void* scale_a_ptr = reinterpret_cast<void*>(scale_a.data_ptr());
    void* scale_b_ptr = reinterpret_cast<void*>(scale_b.data_ptr());

    hipblasLtMatmulDescSetAttribute(matmul_desc,
            HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            reinterpret_cast<void*>(&scale_a_ptr),
            sizeof(void*));
    hipblasLtMatmulDescSetAttribute(matmul_desc,
            HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            reinterpret_cast<void*>(&scale_b_ptr),
            sizeof(void*));

    if (bias.has_value()) {
        hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
        void* bias_ptr = reinterpret_cast<void*>(bias.value().data_ptr());
        hipDataType bias_type = HIP_R_16BF;
        hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue , sizeof(int32_t));
        hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(int32_t));
        hipblasLtMatmulDescSetAttribute(matmul_desc,
                HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                reinterpret_cast<void*>(&bias_ptr),
                sizeof(void*));
    }

    // 设置查询信息
    hipblasLtMatmulHeuristicResult_t heuristicResult[1];
    hipblasLtMatmulPreference_t pref;
    int returnedAlgoCount = 0;
    hipblasLtMatmulPreferenceCreate(&pref);
    int max_workspace_size=0;
    hipblasLtMatmulPreferenceSetAttribute(
            pref,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &max_workspace_size,
            sizeof(max_workspace_size));
    // 自动寻优配置
    auto ret = hipblasLtMatmulAlgoGetHeuristic(
            handle, matmul_desc,
            mat_a_layout, mat_b_layout, mat_c_layout, output_layout,
            pref, 1, heuristicResult, &returnedAlgoCount);
    if (ret != HIPBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
        std::cerr << "[BLASLT] No suitable hipBLASLt algo found, m: " << m << ", n: " << n << ", k: " << k << std::endl;
        return ;
    }
    void* mat_c_fake_ptr = reinterpret_cast<void*>(0x01);
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    hipblasLtMatmul(
            handle, matmul_desc, alpha.data_ptr(),
            mat_a.data_ptr(), mat_a_layout,
            mat_b.data_ptr(), mat_b_layout,
            beta.data_ptr(),
            mat_c_fake_ptr, mat_c_layout,
            output.data_ptr(), output_layout,
            &heuristicResult[0].algo, nullptr, 0, stream);

    hipblasLtMatrixLayoutDestroy(mat_a_layout);
    hipblasLtMatrixLayoutDestroy(mat_b_layout);
    hipblasLtMatrixLayoutDestroy(mat_c_layout);
    hipblasLtMatrixLayoutDestroy(output_layout);
    hipblasLtMatmulPreferenceDestroy(pref);
    return ;
}
} // hipblaslt_gemm