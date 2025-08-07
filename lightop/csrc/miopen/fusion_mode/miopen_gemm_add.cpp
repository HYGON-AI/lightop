#include "miopen/miopen-wrapper.h"
#include "miopen/op/op.h"
#include "miopen/graph/tensor.hpp"
#include "miopen/graph/execution_plan.hpp"
#include "miopen/graph/varpack.hpp"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/ATen.h>

namespace at {
namespace native {
Tensor miopen_gemm_add(const Tensor& input_a,
                       const Tensor& input_b,
                       const Tensor& bias)
{

    if(input_a.dim() != 3 || input_b.dim() != 2 || bias.dim() != 2)
        MIOPEN_THROW("Gemm_Add Do not match input tensor dim");

    const int batch_size = input_a.sizes()[0];
    const int m = input_a.sizes()[1];
    const int n = bias.sizes()[1];

    TensorDescriptor gemm_x1(input_a);
    TensorDescriptor gemm_x2(input_b);
    TensorDescriptor gemm_bias(bias);
    auto data_type = getDataType(input_a);

    // output
    Tensor out = at::empty({batch_size, m, n}, input_a.options());
    TensorDescriptor output_tensor(out);

    // matmul op
    auto [matmul_input1, matmul_input1_uuid] = CreatGraphTensor(gemm_x1, data_type, false);
    auto [matmul_input2, matmul_input2_uuid] = CreatGraphTensor(gemm_x2, data_type, false);
    auto [matmul_output, matmul_output_uuid] = CreatGraphTensor(output_tensor, data_type, true);
    auto [matmul_operation, matmul_operation_desc] =
        CreatGemmOp(data_type, batch_size, matmul_input1, matmul_input2, matmul_output);

    // add op
    auto [add_input, add_input_uuid]   = CreatGraphTensor(gemm_bias, data_type, false);
    auto [add_output, add_output_uuid] = CreatGraphTensor(output_tensor, data_type, false);
    auto [add_operation, add_operation_desc] =
        CreatPointWiseOp(data_type, MIOPEN_POINTWISE_ADD, matmul_output, add_input, add_output);

    // creat plan
    auto exec_plan = CreatExeutionPlan({matmul_operation.desc(), add_operation.desc()});

    // creat varpack
    std::vector<std::pair<size_t, void*>> tensor_id_with_addr;
    tensor_id_with_addr.emplace_back(std::make_pair(matmul_input1_uuid, input_a.data_ptr()));
    tensor_id_with_addr.emplace_back(std::make_pair(matmul_input2_uuid, input_b.data_ptr()));
    tensor_id_with_addr.emplace_back(std::make_pair(add_input_uuid, bias.data_ptr()));
    tensor_id_with_addr.emplace_back(std::make_pair(add_output_uuid, out.data_ptr()));

    size_t workspaceSize = GetPlanWorkspaceSize(exec_plan);

    Tensor workspace_tensor;
    if(workspaceSize > 0)
    {
        workspace_tensor = at::empty({workspaceSize}, input_a.options());
    }
    auto varpack =
        CreatVarpack(tensor_id_with_addr, workspace_tensor.defined() ? workspace_tensor.data_ptr() : nullptr);

    // run kernel
    MIOPEN_CHECK(miopenBackendExecute(getMiopenHandle_new(), exec_plan.desc(), varpack.desc()));

    return out;
}
} // namespace native

} // namespace at
