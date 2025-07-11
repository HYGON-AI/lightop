
#include "miopen/miopen-wrapper.h"
#include "miopen/op/op.h"
#include "miopen/graph/tensor.hpp"
#include "miopen/Handle.h"
#include <vector>
namespace at {
namespace native {

void miopen_fusion_conv_bias_add(miopenDataType_t data_type, miopenTensorLayout_t tensor_layout)
{

    TensorDescriptor conv_input_tensor;
    TensorDescriptor conv_weight_tensor;
    TensorDescriptor conv_output_tensor;
    TensorDescriptor bias_input_tensor;
    TensorDescriptor bias_output_tensor;
    TensorDescriptor add_input_tensor;
    TensorDescriptor add_output_tensor;
    int pads;
    int strides;
    int dilations;
    float alpha;
    float beta;

    std::vector<std::pair<size_t, size_t>> tensor_id_with_size;
    auto [conv_input, conv_input_uuid] =
        CreatGraphTensor(conv_input_tensor, data_type, tensor_layout, false);
    auto [conv_weight, conv_weight_uuid] =
        CreatGraphTensor(conv_weight_tensor, data_type, tensor_layout, false);
    auto [conv_output, conv_output_uuid] =
        CreatGraphTensor(conv_output_tensor, data_type, tensor_layout, true);

    tensor_id_with_size.emplace_back(
        std::make_pair(conv_input_uuid, conv_input_tensor.number_size()));
    tensor_id_with_size.emplace_back(
        std::make_pair(conv_weight_uuid, conv_weight_tensor.number_size()));

    // conv op
    auto conv_operation = CreatConvOp(
        data_type, conv_input, conv_weight, pads, strides, dilations, alpha, beta, conv_output);
    auto [bias_input, bias_input_uuid] =
        CreatGraphTensor(bias_input_tensor, data_type, tensor_layout, false);
    auto [bias_output, bias_output_uuid] =
        CreatGraphTensor(bias_output_tensor, data_type, tensor_layout, true);
    tensor_id_with_size.emplace_back(
        std::make_pair(bias_input_uuid, bias_input_tensor.number_size()));

    // bias op
    auto bias_operation =
        CreatPointWiseOp(data_type, MIOPEN_POINTWISE_ADD, conv_output, bias_input, bias_output);
    auto [add_input, add_input_uuid] =
        CreatGraphTensor(add_input_tensor, data_type, tensor_layout, false);
    auto [add_output, add_output_uuid] =
        CreatGraphTensor(add_output_tensor, data_type, tensor_layout, false);
    tensor_id_with_size.emplace_back(
        std::make_pair(add_input_uuid, add_input_tensor.number_size()));
    tensor_id_with_size.emplace_back(
        std::make_pair(add_output_uuid, add_output_tensor.number_size()));
    // add op
    auto add_operation =
        CreatPointWiseOp(data_type, MIOPEN_POINTWISE_ADD, bias_output, add_input, add_output);
    // creat plan
    auto exec_plan =
        CreatExeutionPlan({conv_operation.desc(), bias_operation.desc(), add_operation.desc()});
    // creat varpack
    auto varpack = CreatVarpack(exec_plan, tensor_id_with_size);
    // run
    miopenBackendExecute(getMiopenHandle_new(), exec_plan.desc(), varpack.desc());
}

} // namespace native
} // namespace at