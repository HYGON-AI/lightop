#include "miopen/miopen-wrapper.h"
#include "miopen/Descriptors.h"

namespace at {
namespace native {

std::pair<BackendDescriptor, size_t> CreatGraphTensor(TensorDescriptor tensor,
                                                      miopenDataType_t data_type,
                                                      miopenTensorLayout_t tensor_layout,
                                                      const bool is_virtual);

} // namespace native
} // namespace at