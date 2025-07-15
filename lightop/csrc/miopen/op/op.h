#pragma once
#include "miopen/graph/tensor.hpp"
#include <vector>

namespace at {
namespace native {
// conv op
std::pair<BackendDescriptor, BackendDescriptor> CreatConvOp(miopenDataType_t data_type,
                                                            const BackendDescriptor& input,
                                                            const BackendDescriptor& weight,
                                                            const std::vector<int64_t>& pads,
                                                            const std::vector<int64_t>& strides,
                                                            const std::vector<int64_t>& dilations,
                                                            const BackendDescriptor& output);
// pointwise op
std::pair<BackendDescriptor, BackendDescriptor> CreatPointWiseOp(miopenDataType_t data_type,
                                                                 miopenPointwiseMode_t mode,
                                                                 const BackendDescriptor& pre_operator_output,
                                                                 const BackendDescriptor& input,
                                                                 const BackendDescriptor& output);

std::pair<BackendDescriptor, BackendDescriptor> CreatPointWiseOp(miopenDataType_t data_type,
                                                                 miopenPointwiseMode_t mode,
                                                                 const BackendDescriptor& pre_operator_output,
                                                                 const BackendDescriptor& output);

} // namespace native
} // namespace at