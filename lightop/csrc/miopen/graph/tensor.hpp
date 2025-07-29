#include "miopen/miopen-wrapper.h"
#include "miopen/Descriptors.h"
#include <ATen/ATen.h>
#include <vector>

namespace at {
namespace native {

std::pair<BackendDescriptor, size_t>
CreatGraphTensor(const TensorDescriptor& tensor, miopenDataType_t data_type, const bool is_virtual);

std::vector<int64_t> GetConvForwardOutputDim(const TensorDescriptor& input,
                                             const TensorDescriptor& weight,
                                             const std::vector<int64_t>& pads,
                                             const std::vector<int64_t>& strides,
                                             const std::vector<int64_t>& dilations);
} // namespace native
} // namespace at