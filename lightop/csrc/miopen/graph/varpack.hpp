#pragma once
#include "miopen/graph/tensor.hpp"
#include <vector>

namespace at {
namespace native {

BackendDescriptor CreatVarpack(BackendDescriptor exec_plan,
                               const std::vector<std::pair<size_t, size_t>> tensor_id_with_size);

} // namespace native
} // namespace at