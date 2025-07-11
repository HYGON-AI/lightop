
#include "miopen/graph/execution_plan.hpp"
#include "miopen/graph/varpack.hpp"
#include <vector>

namespace at {
namespace native {
BackendDescriptor CreatVarpack(BackendDescriptor exec_plan,
                               const std::vector<std::pair<size_t, size_t>> tensor_id_with_size)
{
    std::vector<size_t> ids;
    std::vector<void*> data_ptr;
    for(auto [id, memory_size] : tensor_id_with_size)
    {
        ids.push_back(id);
        void* data;
        auto error = hipMalloc(&data, memory_size);
        if(hipSuccess != error) {}
        data_ptr.push_back(data);
    }

    BackendDescriptor varpack(MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    varpack.set(MIOPEN_ATTR_VARIANT_PACK_DATA_POINTERS,
                MIOPEN_TYPE_VOID_PTR,
                data_ptr.size(),
                data_ptr.data());
    varpack.set(MIOPEN_ATTR_VARIANT_PACK_UNIQUE_IDS, MIOPEN_TYPE_INT64, ids.size(), ids.data());

    auto workspaceSize = GetPlanWorkspaceSize(exec_plan);
    if(workspaceSize > 0)
    {
        void* workspace = nullptr;
        auto error      = hipMalloc(&workspace, workspaceSize);
        if(hipSuccess != error) {}
        varpack.set(MIOPEN_ATTR_VARIANT_PACK_WORKSPACE, MIOPEN_TYPE_VOID_PTR, 1, workspace);
    }
    varpack.finalize();
    return varpack;
}

} // namespace native
} // namespace at