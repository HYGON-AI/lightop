#pragma once
#include "miopen/graph/tensor.hpp"
#include <vector>

namespace at {
namespace native {
// conv op
BackendDescriptor CreatConvOp(miopenDataType_t data_type,
                              const BackendDescriptor& input,
                              const BackendDescriptor& weight,
                              const std::vector<int64_t>& pads,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& dilations,
                              float alpha,
                              float beta,
                              const BackendDescriptor& output);
// pointwise op
BackendDescriptor CreatPointWiseOp(miopenDataType_t data_type,
                                   miopenPointwiseMode_t mode,
                                   const BackendDescriptor& pre_operator_output,
                                   const BackendDescriptor& input,
                                   const BackendDescriptor& output);

} // namespace native
} // namespace at