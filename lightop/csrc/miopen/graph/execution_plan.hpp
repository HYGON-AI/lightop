#pragma once
#include "miopen/graph/tensor.hpp"
#include <vector>

namespace at {
namespace native {

// exec plan
BackendDescriptor CreatExeutionPlan(const std::vector<miopenBackendDescriptor_t>& ops_operation);
size_t GetPlanWorkspaceSize(BackendDescriptor plan)
{
    [[maybe_unused]] int64_t elementCount;
    size_t workspaceSize = 0;

    plan.get(MIOPEN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
             MIOPEN_TYPE_INT64,
             1,
             &elementCount,
             &workspaceSize);
    return workspaceSize;
}

} // namespace native
} // namespace at