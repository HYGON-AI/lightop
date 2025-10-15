#pragma once
#include "miopen/graph/tensor.hpp"
#include <vector>

namespace at {
namespace native {

BackendDescriptor CreatVarpack(const std::vector<std::pair<size_t, void*>> tensor_id_with_addr, void* workspace);

} // namespace native
} // namespace at