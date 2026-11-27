#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <vector>

namespace {
void check(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error("hipBLASLt status " + std::to_string(status));
}

struct Plan {
    std::mutex submission;
    hipblasLtHandle_t handle{};
    hipblasLtMatmulDesc_t desc{};
    hipblasLtMatrixLayout_t a{}, b{}, c{}, d{};
    int32_t alpha = 1, beta = 0;
    std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;

    ~Plan() {
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
        if (d) hipblasLtMatrixLayoutDestroy(d);
        if (desc) hipblasLtMatmulDescDestroy(desc);
        if (handle) hipblasLtDestroy(handle);
    }

    void scales(void* scale_a, void* scale_b) {
        check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                              &scale_a, sizeof(scale_a)));
        check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                              &scale_b, sizeof(scale_b)));
    }

    void init(int m, int n, int k, void* scale_a, void* scale_b, size_t limit) {
        check(hipblasLtCreate(&handle));
        check(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32I, HIP_R_32F));
        auto trans_a = HIPBLAS_OP_T;
        auto trans_b = HIPBLAS_OP_N;
        auto mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                              &trans_a, sizeof(trans_a)));
        check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                              &trans_b, sizeof(trans_b)));
        check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                              &mode, sizeof(mode)));
        check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                              &mode, sizeof(mode)));
        scales(scale_a, scale_b);
        check(hipblasLtMatrixLayoutCreate(&a, HIP_R_8I, k, m, k));
        check(hipblasLtMatrixLayoutCreate(&b, HIP_R_8I, k, n, k));
        check(hipblasLtMatrixLayoutCreate(&c, HIP_R_32I, m, n, m));
        check(hipblasLtMatrixLayoutCreate(&d, HIP_R_16BF, m, n, m));

        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        check(hipblaslt_ext::getAllAlgos(handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
              trans_a, trans_b, HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_16BF,
              HIPBLAS_COMPUTE_32I, results));
        for (auto result : results) {
            size_t workspace = 0;
            auto status = hipblaslt_ext::matmulIsAlgoSupported(
                handle, desc, &alpha, a, b, &beta, c, d, result.algo, workspace);
            if (status == HIPBLAS_STATUS_SUCCESS && workspace <= limit) {
                result.workspaceSize = workspace;
                algorithms.push_back(result);
            }
        }
    }
};
}  // namespace

extern "C" {
void* lightop_minimax31_create_plan(int m, int n, int k, void* scale_a, void* scale_b, size_t workspace) {
    try {
        auto plan = std::make_unique<Plan>();
        plan->init(m, n, k, scale_a, scale_b, workspace);
        return plan.release();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return nullptr;
    }
}
void lightop_minimax31_destroy_plan(Plan* plan) { delete plan; }
int lightop_minimax31_plan_count(Plan* plan) { return plan->algorithms.size(); }
int lightop_minimax31_solution_id(Plan* plan, int index) {
    return hipblaslt_ext::getIndexFromAlgo(plan->algorithms.at(index).algo);
}
size_t lightop_minimax31_solution_workspace(Plan* plan, int index) {
    return plan->algorithms.at(index).workspaceSize;
}
int lightop_minimax31_run_plan(Plan* plan, int index, void* a, void* b, void* scale_a,
             void* scale_b, void* output, void* workspace, hipStream_t stream) {
    try {
        // hipBLASLt consumes the descriptor at submission. Keep both scale
        // pointer updates and GEMM submission atomic for a shared plan.
        const std::lock_guard<std::mutex> lock(plan->submission);
        plan->scales(scale_a, scale_b);
        auto& result = plan->algorithms.at(index);
        return hipblasLtMatmul(plan->handle, plan->desc, &plan->alpha,
              a, plan->a, b, plan->b, &plan->beta, output, plan->c,
              output, plan->d, &result.algo, workspace, result.workspaceSize, stream);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return -1;
    }
}
}
