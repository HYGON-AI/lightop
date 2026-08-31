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
#include <memory>
#include <thread>

namespace hipblaslt_gemm_v2 {

enum class BlockMode
{
    None =0,
    block_128x128x1x128= 1,
    block_64x64x1x64 =2,
};


hipblasOperation_t get_transpose(char trans){
    if(trans == 'N' || trans == 'n')
        return HIPBLAS_OP_N;
    else
        return HIPBLAS_OP_T;
}

class HipBlasLtHandleManager {
public:
    static HipBlasLtHandleManager& instance() {
        static HipBlasLtHandleManager mgr;
        return mgr;
    }

    hipblasLtHandle_t getHandleForCurrentDevice() {
        int dev;
        hipGetDevice(&dev);
        std::call_once(*init_flags[dev], [&](){
            auto status = hipblasLtCreate(&handles[dev]);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("hipblasLtCreate failed");
            }
        });
        return handles[dev];
    }

private:
    HipBlasLtHandleManager() {
        int device_count = 0;
        hipGetDeviceCount(&device_count);
        handles.resize(device_count);
        init_flags.resize(device_count);
        for (int i = 0; i < device_count; i++) {
            init_flags[i] = std::make_unique<std::once_flag>();
        }
    }

    ~HipBlasLtHandleManager() {
        for (auto& h : handles) {
            if (h != nullptr) {
                hipblasLtDestroy(h);
            }
        }
    }

    std::vector<hipblasLtHandle_t> handles;
    std::vector<std::unique_ptr<std::once_flag>> init_flags;
};



class BlasltMatmulDesc {
public:
    BlasltMatmulDesc() = delete;
    BlasltMatmulDesc(hipblasComputeType_t compute_type, hipDataType scale_type) {
        hipblasLtMatmulDescCreate(&desc_, compute_type, scale_type);
    }
    ~BlasltMatmulDesc() {
        hipblasLtMatmulDescDestroy(desc_);
    }
    hipblasLtMatmulDesc_t get() const {
        return desc_;
    }
private:
    hipblasLtMatmulDesc_t desc_;
};

class BlasltMatmulDescManager {
public:
    static BlasltMatmulDesc& get(hipblasComputeType_t compute_type, hipDataType scale_type) {
        using Key = std::pair<int,int>;
        thread_local std::map<Key, std::unique_ptr<BlasltMatmulDesc>> tls_map;
        Key key{static_cast<int>(compute_type), static_cast<int>(scale_type)};
        auto it = tls_map.find(key);
        if (it == tls_map.end()) {
            auto desc = std::make_unique<BlasltMatmulDesc>(compute_type, scale_type);
            auto& ref = *desc;
            tls_map.emplace(key, std::move(desc));
            return ref;
        }
        return *(it->second);
    }
};



hipDataType getDataType(const at::Tensor& t )
{
    switch (t.scalar_type())
    {
        case at::kHalf:           return HIP_R_16F;
        case at::kBFloat16:       return HIP_R_16BF;
        case at::kFloat:          return HIP_R_32F;
        case at::kDouble:         return HIP_R_64F;
        case at::kChar:           return HIP_R_8I;
        case at::kInt:            return HIP_R_32I;
        case at::kFloat8_e4m3fn:  return HIP_R_8F_E4M3;
        case at::kFloat8_e5m2:    return HIP_R_8F_E5M2;
        default:
            std::cerr << "Unsupported tensor dtype for hipDataType" << std::endl;
            std::exit(EXIT_FAILURE);
    }

}

hipDataType getDataType(const at::ScalarType t )
{
    switch (t)
    {
        case at::kHalf:           return HIP_R_16F;
        case at::kBFloat16:       return HIP_R_16BF;
        case at::kFloat:          return HIP_R_32F;
        case at::kDouble:         return HIP_R_64F;
        case at::kChar:           return HIP_R_8I;
        case at::kInt:            return HIP_R_32I;
        case at::kFloat8_e4m3fn:  return HIP_R_8F_E4M3;
        case at::kFloat8_e5m2:    return HIP_R_8F_E5M2;
        default:
            std::cerr << "Unsupported tensor dtype for hipDataType" << std::endl;
            std::exit(EXIT_FAILURE);
    }

}


hipblasComputeType_t getComputeType(const at::Tensor& t)
{
    auto dtype = t.scalar_type();
    if (dtype == at::kFloat)    return HIPBLAS_COMPUTE_32F;
    else if (dtype == at::kInt) return HIPBLAS_COMPUTE_32I;
    else
    {
        std::cerr << "Unsupported tensor dtype for hipblasComputeType" << std::endl;
        std::exit(EXIT_FAILURE);   // 直接终止程序
    }
}


torch::Tensor hipblaslt_gemm_channelwise(const torch::Tensor& d_A, const torch::Tensor& d_B,
        const torch::Tensor& scale_a, const torch::Tensor& scale_b,
        at::ScalarType out_dtype,
        const int64_t m, const int64_t n, const int64_t k, const int64_t batch,
        const std::string& transpose_flag,
        const torch::Tensor& alpha, const torch::Tensor& beta,
        const std::optional<torch::Tensor>& bias) {

    hipblasLtHandle_t handle = HipBlasLtHandleManager::instance().getHandleForCurrentDevice();

     std::vector<long> size_D{batch, n, m};
     torch::Tensor output=at::empty(size_D,d_A.options().dtype(out_dtype));

    hipblasOperation_t trans_a = get_transpose(transpose_flag[0]);
    hipblasOperation_t trans_b = get_transpose(transpose_flag[1]);

    //制作入参
    size_t A_row = trans_a == HIPBLAS_OP_N ? m : k;
    size_t A_col = trans_a == HIPBLAS_OP_N ? k : m;
    size_t B_row = trans_b == HIPBLAS_OP_N ? k : n;
    size_t B_col = trans_b == HIPBLAS_OP_N ? n : k;

    hipDataType A_type=getDataType(d_A);
    hipDataType B_type=getDataType(d_B);
    hipDataType C_type=getDataType(out_dtype);
    hipDataType D_type=getDataType(out_dtype);
    hipblasComputeType_t compute_type=getComputeType(alpha);
    hipDataType scale_type=getDataType(scale_a);

    if(A_type == HIP_R_8I &&  B_type == HIP_R_8I ) {
        C_type= HIP_R_32I;
    }

    int64_t stride_a = m * k;
    int64_t stride_b = n * k;
    int64_t stride_c = m * n;
    int64_t stride_d = m * n;

    size_t lda = trans_a==HIPBLAS_OP_N ? m : k;
    size_t ldb = trans_b==HIPBLAS_OP_N ? k : n;
    size_t ldc = m;
    size_t ldd = m;

    // 获取desc
    auto& desc = BlasltMatmulDescManager::get(compute_type, scale_type);
    hipblasLtMatmulDesc_t matmul_desc = desc.get();


    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    // 设置输入输出矩阵的类型
    hipblasLtMatrixLayoutCreate(&matA, A_type, A_row, A_col, lda);
    hipblasLtMatrixLayoutCreate(&matB, B_type, B_row, B_col, ldb);
    hipblasLtMatrixLayoutCreate(&matC, C_type, m, n, ldc);
    hipblasLtMatrixLayoutCreate(&matD, D_type, m, n, ldd);

    // 处理多batch
    if(batch>1) {
        hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &(stride_a), sizeof(int64_t));

        hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &(stride_b), sizeof(int64_t));

        hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &(stride_c), sizeof(int64_t));

        hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &(stride_d), sizeof(int64_t));
    }

    void* scale_a_ptr = reinterpret_cast<void*>(scale_a.data_ptr());
    void* scale_b_ptr = reinterpret_cast<void*>(scale_b.data_ptr());

    // 设置计算属性
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t));
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t));

    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            reinterpret_cast<void*>(&scale_a_ptr), sizeof(void*));
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            reinterpret_cast<void*>(&scale_b_ptr), sizeof(void*));

    hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, (void*)&mode, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, (void*)&mode, sizeof(void*));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

    if (bias.has_value()) {
        epilogue = HIPBLASLT_EPILOGUE_BIAS;
        void* bias_ptr = reinterpret_cast<void*>(bias.value().data_ptr());
        hipDataType  bias_type  = HIP_R_16F; //getDataType(bias.value());
        hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                &bias_type, sizeof(int32_t));
        hipblasLtMatmulDescSetAttribute(matmul_desc,
                HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                reinterpret_cast<void*>(&bias_ptr),
                sizeof(void*));
    }
    // 防止复用前方的bias属性
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(int32_t));

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
    auto status = hipblasLtMatmulAlgoGetHeuristic(
            handle, matmul_desc,
            matA, matB, matC, matD,
            pref, 1, heuristicResult, &returnedAlgoCount);
    if (status != HIPBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
        std::cerr << "[BLASLT] No suitable hipBLASLt algo found, m: "
                << m << ", n: " << n << ", k: " << k << " status: "<<status<<std::endl;
    }

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    status=hipblasLtMatmul(
            handle, matmul_desc, alpha.data_ptr(),
            d_A.data_ptr(), matA,
            d_B.data_ptr(), matB,
            beta.data_ptr(),
            output.data_ptr(), matC,
            output.data_ptr(), matD,
            &heuristicResult[0].algo, nullptr, 0, stream);
    if (status != HIPBLAS_STATUS_SUCCESS ) {
        std::cerr << "[BLASLT] hipblasLtMatmul  failed , m: "
                << m << ", n: " << n << ", k: " << k << " status: "<<status<<std::endl;
    }
    hipblasLtMatrixLayoutDestroy(matA);
    hipblasLtMatrixLayoutDestroy(matB);
    hipblasLtMatrixLayoutDestroy(matC);
    hipblasLtMatrixLayoutDestroy(matD);
    hipblasLtMatmulPreferenceDestroy(pref);
    return output;
}

// KME/gfx928 int8_bf16_cw: C/D=bf16, compute=fp32 (see src-check/main.cpp int8_bf16_cw).
// Same API as hipblaslt_gemm_channelwise; does not force int8×int8 C matrix to int32.
torch::Tensor hipblaslt_gemm_channelwise_kme(const torch::Tensor& d_A, const torch::Tensor& d_B,
        const torch::Tensor& scale_a, const torch::Tensor& scale_b,
        at::ScalarType out_dtype,
        const int64_t m, const int64_t n, const int64_t k, const int64_t batch,
        const std::string& transpose_flag,
        const torch::Tensor& alpha, const torch::Tensor& beta,
        const std::optional<torch::Tensor>& bias) {

    hipblasLtHandle_t handle = HipBlasLtHandleManager::instance().getHandleForCurrentDevice();

     std::vector<long> size_D{batch, n, m};
     torch::Tensor output=at::empty(size_D,d_A.options().dtype(out_dtype));

    hipblasOperation_t trans_a = get_transpose(transpose_flag[0]);
    hipblasOperation_t trans_b = get_transpose(transpose_flag[1]);

    size_t A_row = trans_a == HIPBLAS_OP_N ? m : k;
    size_t A_col = trans_a == HIPBLAS_OP_N ? k : m;
    size_t B_row = trans_b == HIPBLAS_OP_N ? k : n;
    size_t B_col = trans_b == HIPBLAS_OP_N ? n : k;

    hipDataType A_type=getDataType(d_A);
    hipDataType B_type=getDataType(d_B);
    hipDataType C_type=getDataType(out_dtype);
    hipDataType D_type=getDataType(out_dtype);
    hipblasComputeType_t compute_type=HIPBLAS_COMPUTE_32F;
    hipDataType scale_type=getDataType(scale_a);

    int64_t stride_a = m * k;
    int64_t stride_b = n * k;
    int64_t stride_c = m * n;
    int64_t stride_d = m * n;

    size_t lda = trans_a==HIPBLAS_OP_N ? m : k;
    size_t ldb = trans_b==HIPBLAS_OP_N ? k : n;
    size_t ldc = m;
    size_t ldd = m;

    auto& desc = BlasltMatmulDescManager::get(compute_type, scale_type);
    hipblasLtMatmulDesc_t matmul_desc = desc.get();

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    hipblasLtMatrixLayoutCreate(&matA, A_type, A_row, A_col, lda);
    hipblasLtMatrixLayoutCreate(&matB, B_type, B_row, B_col, ldb);
    hipblasLtMatrixLayoutCreate(&matC, C_type, m, n, ldc);
    hipblasLtMatrixLayoutCreate(&matD, D_type, m, n, ldd);

    if(batch>1) {
        hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &(stride_a), sizeof(int64_t));

        hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &(stride_b), sizeof(int64_t));

        hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &(stride_c), sizeof(int64_t));

        hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &(stride_d), sizeof(int64_t));
    }

    void* scale_a_ptr = reinterpret_cast<void*>(scale_a.data_ptr());
    void* scale_b_ptr = reinterpret_cast<void*>(scale_b.data_ptr());

    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t));
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t));

    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            reinterpret_cast<void*>(&scale_a_ptr), sizeof(void*));
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            reinterpret_cast<void*>(&scale_b_ptr), sizeof(void*));

    hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, (void*)&mode, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, (void*)&mode, sizeof(void*));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

    if (bias.has_value()) {
        epilogue = HIPBLASLT_EPILOGUE_BIAS;
        void* bias_ptr = reinterpret_cast<void*>(bias.value().data_ptr());
        hipDataType  bias_type  = getDataType(out_dtype);
        hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                &bias_type, sizeof(int32_t));
        hipblasLtMatmulDescSetAttribute(matmul_desc,
                HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                reinterpret_cast<void*>(&bias_ptr),
                sizeof(void*));
    }
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(int32_t));

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

    auto status = hipblasLtMatmulAlgoGetHeuristic(
            handle, matmul_desc,
            matA, matB, matC, matD,
            pref, 1, heuristicResult, &returnedAlgoCount);
    if (status != HIPBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
        std::cerr << "[BLASLT-KME] No suitable hipBLASLt algo found, m: "
                << m << ", n: " << n << ", k: " << k << " status: "<<status<<std::endl;
    }

    const float alpha_f = alpha.scalar_type() == at::ScalarType::Float
            ? alpha.item<float>() : static_cast<float>(alpha.item<int>());
    const float beta_f = beta.scalar_type() == at::ScalarType::Float
            ? beta.item<float>() : static_cast<float>(beta.item<int>());

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    status=hipblasLtMatmul(
            handle, matmul_desc, &alpha_f,
            d_A.data_ptr(), matA,
            d_B.data_ptr(), matB,
            &beta_f,
            output.data_ptr(), matC,
            output.data_ptr(), matD,
            &heuristicResult[0].algo, nullptr, 0, stream);
    if (status != HIPBLAS_STATUS_SUCCESS ) {
        std::cerr << "[BLASLT-KME] hipblasLtMatmul failed, m: "
                << m << ", n: " << n << ", k: " << k << " status: "<<status<<std::endl;
    }
    hipblasLtMatrixLayoutDestroy(matA);
    hipblasLtMatrixLayoutDestroy(matB);
    hipblasLtMatrixLayoutDestroy(matC);
    hipblasLtMatrixLayoutDestroy(matD);
    hipblasLtMatmulPreferenceDestroy(pref);
    return output;
}

torch::Tensor hipblaslt_gemm_channelwise_fast(const torch::Tensor& d_A, const torch::Tensor& d_B,
        const torch::Tensor& scale_a, const torch::Tensor& scale_b,
        at::ScalarType out_dtype,
        const int64_t m, const int64_t n, const int64_t k, const int64_t batch,
        const std::string& transpose_flag,
        const torch::Tensor& alpha, const torch::Tensor& beta,
        const std::optional<torch::Tensor>& bias) {

    hipblasLtHandle_t handle = HipBlasLtHandleManager::instance().getHandleForCurrentDevice();
    std::vector<long> size_D{batch, n, m};
    torch::Tensor output=at::empty(size_D,d_A.options().dtype(out_dtype));

    hipblasOperation_t trans_a = get_transpose(transpose_flag[0]);
    hipblasOperation_t trans_b = get_transpose(transpose_flag[1]);


    hipDataType A_type=getDataType(d_A);
    hipDataType B_type=getDataType(d_B);
    hipDataType C_type=getDataType(out_dtype);
    hipDataType D_type=getDataType(out_dtype);
    hipblasComputeType_t compute_type=getComputeType(alpha);
    hipDataType scale_type=getDataType(scale_a);
    hipDataType bias_type=D_type;

    if(A_type == HIP_R_8I &&  B_type == HIP_R_8I){
        C_type= HIP_R_32I;
    }

    size_t lda = trans_a==HIPBLAS_OP_N ? m : k;
    size_t ldb = trans_b==HIPBLAS_OP_N ? k : n;
    size_t ldc = m;
    size_t ldd = m;

    void* scale_a_ptr = reinterpret_cast<void*>(scale_a.data_ptr());
    void* scale_b_ptr = reinterpret_cast<void*>(scale_b.data_ptr());

    hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    void* bias_ptr=nullptr;
    if (bias.has_value()) {
         bias_ptr = reinterpret_cast<void*>(bias.value().data_ptr());
         bias_type=getDataType(bias.value());
    }
    hipblasStatus_t status;
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if(batch==1)
    {
        status= hipblasLtGemmExQ( handle,
                                    trans_a,trans_b,
                                    m,n,k,
                                    alpha.data_ptr(),
                                    d_A.data_ptr(),A_type,lda,
                                    d_B.data_ptr(),B_type,ldb,
                                    beta.data_ptr(),
                                    output.data_ptr(),C_type,ldc,
                                    output.data_ptr(),D_type,ldd,
                                    compute_type,
                                    scale_a_ptr,mode,
                                    scale_b_ptr,mode,
                                    bias_ptr,bias_type,stream);
    }
    else{
        int64_t stride_a = m * k;
        int64_t stride_b = n * k;
        int64_t stride_c = m * n;
        int64_t stride_d = m * n;
        status=hipblasLtStridedBatchedGemmExQ( handle,
                trans_a,trans_b,
                m,n,k,
                alpha.data_ptr(),
                d_A.data_ptr(),A_type,lda,stride_a,
                d_B.data_ptr(),B_type,ldb,stride_b,
                beta.data_ptr(),
                output.data_ptr(),C_type,ldc,stride_c,
                output.data_ptr(),D_type,ldd,stride_d,
                batch,
                compute_type,
                scale_a_ptr,mode,
                scale_b_ptr,mode,
                bias_ptr,bias_type,stream);

    }
    if(status != HIPBLAS_STATUS_SUCCESS){
        std::cerr << "[BLASLT] hipblasLtGemmExQ or hipblasLtStridedBatchedGemmExQ failed , m: "
                << m << ", n: " << n << ", k: " << k << " status: "<<status<<std::endl;
    }

    return output;
}

torch::Tensor hipblaslt_gemm_blockwise(const torch::Tensor& d_A, const torch::Tensor& d_B,
        const torch::Tensor& scale_a, const torch::Tensor& scale_b,
        at::ScalarType out_dtype,
        const int64_t m, const int64_t n, const int64_t k, const int64_t batch,
        const std::string& transpose_flag,const int block_flag,
        const torch::Tensor& alpha, const torch::Tensor& beta,
        const std::optional<torch::Tensor>& bias) {

    hipblasLtHandle_t handle = HipBlasLtHandleManager::instance().getHandleForCurrentDevice();
    std::vector<long> size_D{batch, n, m};
    torch::Tensor output=at::empty(size_D,d_A.options().dtype(out_dtype));
    hipblasOperation_t trans_a = get_transpose(transpose_flag[0]);
    hipblasOperation_t trans_b = get_transpose(transpose_flag[1]);
    //制作入参
    size_t A_row = (trans_a == HIPBLAS_OP_N ? m : k);
    size_t A_col = (trans_a == HIPBLAS_OP_N ? k : m);
    size_t B_row = (trans_b == HIPBLAS_OP_N ? k : n);
    size_t B_col = (trans_b == HIPBLAS_OP_N ? n : k);
    hipDataType A_type=getDataType(d_A);
    hipDataType B_type=getDataType(d_B);
    hipDataType C_type=getDataType(out_dtype);
    hipDataType D_type=getDataType(out_dtype);
    hipblasComputeType_t compute_type=getComputeType(alpha);
    hipDataType scale_type=getDataType(scale_a);
    if(A_type == HIP_R_8I &&  B_type == HIP_R_8I ){
        C_type= HIP_R_32I;
    }


    if (batch!=1)
    {
        std::cerr<<"[BLASLT] blockwise does not currently support batch > 1"<<std::endl;
        return output;
    }
    if (trans_a != HIPBLAS_OP_N || trans_b != HIPBLAS_OP_N)
    {
        std::cerr<<"[BLASLT] blockwise Only supports transpose type NN"<<std::endl;
        return output;
    }
    if (A_type != HIP_R_8F_E4M3 || B_type  != HIP_R_8F_E4M3 )
    {
        std::cerr<<"[BLASLT] blockwise Only supports datatype fp8"<<std::endl;
        return output;
    }

    int64_t stride_a = m * k;
    int64_t stride_b = n * k;
    int64_t stride_c = m * n;
    int64_t stride_d = m * n;

    size_t lda = (trans_a == HIPBLAS_OP_N ? m : k);
    size_t ldb = (trans_b == HIPBLAS_OP_N ? k : n);
    size_t ldc = m;
    size_t ldd = m;

    // 获取desc
    auto& desc = BlasltMatmulDescManager::get(compute_type, scale_type);
    hipblasLtMatmulDesc_t matmul_desc = desc.get();


    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    // 设置输入输出矩阵的类型
    hipblasLtMatrixLayoutCreate(&matA, A_type, A_row, A_col, lda);
    hipblasLtMatrixLayoutCreate(&matB, B_type, B_row, B_col, ldb);
    hipblasLtMatrixLayoutCreate(&matC, C_type, m, n, ldc);
    hipblasLtMatrixLayoutCreate(&matD, D_type, m, n, ldd);

    // 处理多batch
    if(batch>1)
    {
        hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &(stride_a), sizeof(int64_t));

        hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &(stride_b), sizeof(int64_t));

        hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &(stride_c), sizeof(int64_t));

        hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch, sizeof(int32_t));
        hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &(stride_d), sizeof(int64_t));
    }

    void* scale_a_ptr = reinterpret_cast<void*>(scale_a.data_ptr());
    void* scale_b_ptr = reinterpret_cast<void*>(scale_b.data_ptr());

    // 设置计算属性
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t));
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t));

    hipblasLtMatmulDescSetAttribute(
            matmul_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            reinterpret_cast<void*>(&scale_a_ptr), sizeof(void*));
    hipblasLtMatmulDescSetAttribute(
            matmul_desc,HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            reinterpret_cast<void*>(&scale_b_ptr), sizeof(void*));

    hipblasLtMatmulMatrixScale_t modeA ;
    hipblasLtMatmulMatrixScale_t modeB ;
    switch(static_cast<BlockMode>(block_flag))
    {
        case BlockMode::block_128x128x1x128:
            modeA=HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
            modeB=HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
            break;
        case BlockMode::block_64x64x1x64:
            modeA=HIPBLASLT_MATMUL_MATRIX_SCALE_BLK64x64_32F;
            modeB=HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_32F;
            break;
        default:
            std::cerr<<"[BLASLT] This block is currently not supported"<<std::endl;
            return output;

    }


    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, (void*)&modeA, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, (void*)&modeB, sizeof(void*));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

    if (bias.has_value()) {
        epilogue = HIPBLASLT_EPILOGUE_BIAS;
        void* bias_ptr = reinterpret_cast<void*>(bias.value().data_ptr());
        hipDataType  bias_type  = getDataType(bias.value());
        hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(int32_t));
        hipblasLtMatmulDescSetAttribute(matmul_desc,
                HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                reinterpret_cast<void*>(&bias_ptr),
                sizeof(void*));
    }
    // 防止复用前方的bias属性
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(int32_t));

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
    auto status = hipblasLtMatmulAlgoGetHeuristic(
            handle, matmul_desc,
            matA, matB, matC, matD,
            pref, 1, heuristicResult, &returnedAlgoCount);
    if (status != HIPBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
        std::cerr << "[BLASLT] No suitable hipBLASLt algo found, m: " << m << ", n: " << n << ", k: " << k << " status: "<<status<<std::endl;
    }

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    status=hipblasLtMatmul(
            handle, matmul_desc, alpha.data_ptr(),
            d_A.data_ptr(), matA,
            d_B.data_ptr(), matB,
            beta.data_ptr(),
            output.data_ptr(), matC,
            output.data_ptr(), matD,
            &heuristicResult[0].algo, nullptr, 0, stream);
    if (status != HIPBLAS_STATUS_SUCCESS ) {
        std::cerr << "[BLASLT] hipblasLtMatmul  failed , m: " << m << ", n: " << n << ", k: " << k << " status: "<<status<<std::endl;
    }
    hipblasLtMatrixLayoutDestroy(matA);
    hipblasLtMatrixLayoutDestroy(matB);
    hipblasLtMatrixLayoutDestroy(matC);
    hipblasLtMatrixLayoutDestroy(matD);
    hipblasLtMatmulPreferenceDestroy(pref);
    return output;
}

torch::Tensor hipblaslt_gemm_blockwise_fast(const torch::Tensor& d_A, const torch::Tensor& d_B,
        const torch::Tensor& scale_a, const torch::Tensor& scale_b,
        at::ScalarType out_dtype,
        const int64_t m, const int64_t n, const int64_t k, const int64_t batch,
        const std::string& transpose_flag,const int block_flag,
        const torch::Tensor& alpha, const torch::Tensor& beta,
        const std::optional<torch::Tensor>& bias) {

    hipblasLtHandle_t handle = HipBlasLtHandleManager::instance().getHandleForCurrentDevice();
    std::vector<long> size_D{batch, n, m};
    torch::Tensor output=at::empty(size_D,d_A.options().dtype(out_dtype));
    hipblasOperation_t trans_a = get_transpose(transpose_flag[0]);
    hipblasOperation_t trans_b = get_transpose(transpose_flag[1]);
    hipDataType A_type=getDataType(d_A);
    hipDataType B_type=getDataType(d_B);
    hipDataType C_type=getDataType(out_dtype);
    hipDataType D_type=getDataType(out_dtype);
    hipblasComputeType_t compute_type=getComputeType(alpha);
    hipDataType scale_type=getDataType(scale_a);
    hipDataType bias_type=D_type;
    if(A_type == HIP_R_8I &&  B_type == HIP_R_8I){
        C_type= HIP_R_32I;
    }

    if (batch!=1)
    {
        std::cerr<<"[BLASLT] blockwise does not currently support batch > 1"<<std::endl;
        return output;
    }
    if (trans_a != HIPBLAS_OP_N || trans_b != HIPBLAS_OP_N)
    {
        std::cerr<<"[BLASLT] blockwise Only supports transpose type NN"<<std::endl;
        return output;
    }
    if (A_type != HIP_R_8F_E4M3 || B_type  != HIP_R_8F_E4M3 )
    {
        std::cerr<<"[BLASLT] blockwise Only supports datatype fp8"<<std::endl;
        return output;
    }


    size_t lda = trans_a==HIPBLAS_OP_N ? m : k;
    size_t ldb = trans_b==HIPBLAS_OP_N ? k : n;
    size_t ldc = m;
    size_t ldd = m;

    void* scale_a_ptr = reinterpret_cast<void*>(scale_a.data_ptr());
    void* scale_b_ptr = reinterpret_cast<void*>(scale_b.data_ptr());

    hipblasLtMatmulMatrixScale_t modeA ;
    hipblasLtMatmulMatrixScale_t modeB ;
    switch(static_cast<BlockMode>(block_flag))
    {
        case BlockMode::block_128x128x1x128:
            modeA=HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
            modeB=HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
            break;
        case BlockMode::block_64x64x1x64:
            modeA=HIPBLASLT_MATMUL_MATRIX_SCALE_BLK64x64_32F;
            modeB=HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_32F;
            break;
        default:
            std::cerr<<"[BLASLT] This block is currently not supported"<<std::endl;
            return output;

    }


    void* bias_ptr=nullptr;
    if (bias.has_value()) {
         bias_ptr = reinterpret_cast<void*>(bias.value().data_ptr());
         bias_type=getDataType(bias.value());
    }
    hipblasStatus_t status;
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if(batch==1)
    {
        status= hipblasLtGemmExQ( handle,
                                    trans_a,trans_b,
                                    m,n,k,
                                    alpha.data_ptr(),
                                    d_A.data_ptr(),A_type,lda,
                                    d_B.data_ptr(),B_type,ldb,
                                    beta.data_ptr(),
                                    output.data_ptr(),C_type,ldc,
                                    output.data_ptr(),D_type,ldd,
                                    compute_type,
                                    scale_a_ptr,modeA,
                                    scale_b_ptr,modeB,
                                    bias_ptr,bias_type,stream);
    }
    else{
        int64_t stride_a = m * k;
        int64_t stride_b = n * k;
        int64_t stride_c = m * n;
        int64_t stride_d = m * n;
        status=hipblasLtStridedBatchedGemmExQ( handle,
                trans_a,trans_b,
                m,n,k,
                alpha.data_ptr(),
                d_A.data_ptr(),A_type,lda,stride_a,
                d_B.data_ptr(),B_type,ldb,stride_b,
                beta.data_ptr(),
                output.data_ptr(),C_type,ldc,stride_c,
                output.data_ptr(),D_type,ldd,stride_d,
                batch,
                compute_type,
                scale_a_ptr,modeA,
                scale_b_ptr,modeB,
                bias_ptr,bias_type,stream);

    }
    if(status != HIPBLAS_STATUS_SUCCESS){
        std::cerr << "[BLASLT] hipblasLtGemmExQ or hipblasLtStridedBatchedGemmExQ failed , m: " << m << ", n: " << n << ", k: " << k << " status: "<<status<<std::endl;
    }

    return output;
}


}