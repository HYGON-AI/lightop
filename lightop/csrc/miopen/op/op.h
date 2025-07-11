#pragma once
#include "miopen/graph/tensor.hpp"
#include <vector>

namespace at {
namespace native {
// conv op
BackendDescriptor CreatConvOp(miopenDataType_t data_type,
                              BackendDescriptor input,
                              BackendDescriptor weight,
                              std::vector<int64_t> pads,
                              std::vector<int64_t> strides,
                              std::vector<int64_t> dilations,
                              float alpha,
                              float beta,
                              BackendDescriptor output);
// pointwise op
BackendDescriptor CreatPointWiseOp(miopenDataType_t data_type,
                                   miopenPointwiseMode_t mode,
                                   BackendDescriptor pre_operator_output,
                                   BackendDescriptor input,
                                   BackendDescriptor output);

} // namespace native
} // namespace at