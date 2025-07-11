
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
#include <tuple>
namespace at {
namespace native {
Tensor miopen_fusion_conv_bias_add(const Tensor& x,
                                   const Tensor& weight,
                                   const Tensor& bias,
                                   const Tensor& add,
                                   const std::vector<int64_t>& pads,
                                   const std::vector<int64_t>& strides,
                                   const std::vector<int64_t>& dilations,
                                   const std::string& layout,
                                   float alpha,
                                   float beta)

{

    TensorDescriptor conv_input_tensor(x);
    TensorDescriptor conv_weight_tensor(weight);
    TensorDescriptor bias_input_tensor(bias);
    TensorDescriptor add_input_tensor(add);
    auto data_type     = getDataType(x);
    auto tensor_layout = getLayout(layout);

    // output
    auto output_dims = GetConvForwardOutputDim(conv_input_tensor, conv_weight_tensor, pads, strides, dilations);
    Tensor out       = at::empty({output_dims[0], output_dims[1], output_dims[2], output_dims[3]}, x.options());
    TensorDescriptor output_tensor(out);

    // conv op
    auto [conv_input, conv_input_uuid]   = CreatGraphTensor(conv_input_tensor, data_type, tensor_layout, false);
    auto [conv_weight, conv_weight_uuid] = CreatGraphTensor(conv_weight_tensor, data_type, tensor_layout, false);
    auto [conv_output, conv_output_uuid] = CreatGraphTensor(output_tensor, data_type, tensor_layout, true);

    auto conv_operation =
        CreatConvOp(data_type, conv_input, conv_weight, pads, strides, dilations, alpha, beta, conv_output);

    // bias op
    auto [bias_input, bias_input_uuid]   = CreatGraphTensor(bias_input_tensor, data_type, tensor_layout, false);
    auto [bias_output, bias_output_uuid] = CreatGraphTensor(output_tensor, data_type, tensor_layout, true);
    auto bias_operation = CreatPointWiseOp(data_type, MIOPEN_POINTWISE_ADD, conv_output, bias_input, bias_output);
    // add op
    auto [add_input, add_input_uuid]   = CreatGraphTensor(add_input_tensor, data_type, tensor_layout, false);
    auto [add_output, add_output_uuid] = CreatGraphTensor(output_tensor, data_type, tensor_layout, false);
    auto add_operation = CreatPointWiseOp(data_type, MIOPEN_POINTWISE_ADD, bias_output, add_input, add_output);
    // creat plan
    auto exec_plan = CreatExeutionPlan({conv_operation.desc(), bias_operation.desc(), add_operation.desc()});
    // creat varpack
    std::vector<std::pair<size_t, void*>> tensor_id_with_addr;
    tensor_id_with_addr.emplace_back(std::make_pair(conv_input_uuid, x.data_ptr<void*>()));
    tensor_id_with_addr.emplace_back(std::make_pair(conv_weight_uuid, weight.data_ptr<void*>()));
    tensor_id_with_addr.emplace_back(std::make_pair(bias_input_uuid, bias.data_ptr<void*>()));
    tensor_id_with_addr.emplace_back(std::make_pair(add_input_uuid, add.data_ptr<void*>()));
    tensor_id_with_addr.emplace_back(std::make_pair(add_output_uuid, out.data_ptr<void*>()));

    size_t workspaceSize = GetPlanWorkspaceSize(exec_plan);

    Tensor workspace_tensor;
    if(workspaceSize > 0)
    {
        workspace_tensor = at::empty({workspaceSize}, x.options());
    }
    auto varpack =
        CreatVarpack(tensor_id_with_addr, workspace_tensor.defined() ? workspace_tensor.data_ptr<void*>() : nullptr);

    // run kernel
    MIOPEN_CHECK(miopenBackendExecute(getMiopenHandle_new(), exec_plan.desc(), varpack.desc()));

    return out;
}

} // namespace native
} // namespace at