// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace lightop_int8_blaslt {

void check(hipblasStatus_t status) {
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS,
                "LightOp INT8 BLASLt failed with status ", static_cast<int>(status));
}

struct Plan {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
    hipblasLtMatmulAlgo_t algo;
    int32_t alpha = 1, beta = 0;

    Plan(int64_t m, int64_t n, int64_t k, int solution, void* sa, void* sb) {
        check(hipblasLtCreate(&handle));
        check(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32I, HIP_R_32I));
        hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
        check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        auto mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
        check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
        check(hipblasLtMatrixLayoutCreate(&a, HIP_R_8I, k, n, k));
        check(hipblasLtMatrixLayoutCreate(&b, HIP_R_8I, k, m, k));
        check(hipblasLtMatrixLayoutCreate(&c, HIP_R_32I, n, m, n));
        check(hipblasLtMatrixLayoutCreate(&d, HIP_R_16BF, n, m, n));
        check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)));
        check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)));
        std::vector<int> indexes{solution};
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        auto status = hipblaslt_ext::getAlgosFromIndex(handle, indexes, results);
        size_t workspace = 0;
        bool supported = status == HIPBLAS_STATUS_SUCCESS && !results.empty();
        if (supported) {
            algo = results[0].algo;
            supported = hipblaslt_ext::matmulIsAlgoSupported(
                handle, desc, &alpha, a, b, &beta, c, d, algo, workspace)
                == HIPBLAS_STATUS_SUCCESS && workspace == 0;
        }
        if (!supported) {
            hipblasLtMatmulPreference_t pref;
            check(hipblasLtMatmulPreferenceCreate(&pref));
            size_t limit = 0;
            check(hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
            hipblasLtMatmulHeuristicResult_t result;
            int count = 0;
            status = hipblasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, d, pref, 1, &result, &count);
            hipblasLtMatmulPreferenceDestroy(pref);
            check(status);
            TORCH_CHECK(count > 0, "No zero-workspace INT8 BLASLt algorithm");
            algo = result.algo;
        }
    }

    ~Plan() {
        hipblasLtMatrixLayoutDestroy(a);
        hipblasLtMatrixLayoutDestroy(b);
        hipblasLtMatrixLayoutDestroy(c);
        hipblasLtMatrixLayoutDestroy(d);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(handle);
    }
};

std::optional<torch::Tensor> scaled_mm(const torch::Tensor& input, const torch::Tensor& weight,
                                     const torch::Tensor& scale_input, const torch::Tensor& scale_weight,
                                     int solution) {
    if (!(input.is_cuda() && weight.is_cuda() && scale_input.is_cuda() && scale_weight.is_cuda()
          && input.device() == weight.device() && input.device() == scale_input.device()
          && input.device() == scale_weight.device()
          && input.scalar_type() == at::kChar && weight.scalar_type() == at::kChar
          && scale_input.scalar_type() == at::kFloat && scale_weight.scalar_type() == at::kFloat
          && input.dim() == 2 && weight.dim() == 2 && input.is_contiguous()))
        return std::nullopt;
    const auto m = input.size(0), k = input.size(1), n = weight.size(1);
    if (!(weight.size(0) == k && weight.stride(0) == 1 && weight.stride(1) == k
          && scale_input.is_contiguous() && scale_weight.is_contiguous()
          && scale_input.dim() == 2 && scale_weight.dim() == 2
          && scale_input.size(0) == m && scale_input.size(1) == 1
          && scale_weight.size(0) == n && scale_weight.size(1) == 1))
        return std::nullopt;
    const c10::cuda::CUDAGuard guard(input.device());
    static thread_local std::map<int, bool> devices;
    auto device = devices.find(input.get_device());
    if (device == devices.end()) {
        hipDeviceProp_t properties;
        TORCH_CHECK(hipGetDeviceProperties(&properties, input.get_device()) == hipSuccess);
        std::string arch(properties.gcnArchName);
        arch = arch.substr(0, arch.find(':'));
        device = devices.emplace(input.get_device(), arch == "gfx936" && properties.multiProcessorCount == 80).first;
    }
    if (!device->second)
        return std::nullopt;
    using Key = std::tuple<int, int64_t, int64_t, int64_t, int>;
    // Each thread owns its mutable descriptors; every device and shape has a plan.
    static thread_local std::map<Key, std::unique_ptr<Plan>> plans;
    Key key(input.get_device(), m, n, k, solution);
    void* sa = scale_weight.data_ptr();
    void* sb = scale_input.data_ptr();
    auto it = plans.find(key);
    if (it == plans.end())
        it = plans.emplace(key, std::make_unique<Plan>(m, n, k, solution, sa, sb)).first;
    auto& plan = *it->second;
    check(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)));
    check(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)));
    auto output = torch::empty({m, n}, input.options().dtype(at::kBFloat16));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // beta=0: the INT32 C descriptor needs no intermediate tensor or read.
    check(hipblasLtMatmul(plan.handle, plan.desc, &plan.alpha,
                         weight.data_ptr(), plan.a, input.data_ptr(), plan.b,
                         &plan.beta, reinterpret_cast<void*>(0x01), plan.c,
                         output.data_ptr(), plan.d, &plan.algo, nullptr, 0, stream));
    return output;
}

}  // namespace lightop_int8_blaslt

