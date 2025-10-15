
#include "miopen/graph/varpack.hpp"
#include <cstddef>
#include <vector>

namespace at {
namespace native {

BackendDescriptor CreatVarpack(const std::vector<std::pair<size_t, void*>> tensor_id_with_addr, void* workspace)
{
    std::vector<size_t> ids;
    std::vector<void*> data_ptr;
    for(auto [id, addr] : tensor_id_with_addr)
    {
        ids.push_back(id);
        data_ptr.push_back(addr);
    }

    BackendDescriptor varpack(MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    varpack.set(MIOPEN_ATTR_VARIANT_PACK_DATA_POINTERS, MIOPEN_TYPE_VOID_PTR, data_ptr.size(), data_ptr.data());
    varpack.set(MIOPEN_ATTR_VARIANT_PACK_UNIQUE_IDS, MIOPEN_TYPE_INT64, ids.size(), ids.data());

    if(workspace)
    {
        varpack.set(MIOPEN_ATTR_VARIANT_PACK_WORKSPACE, MIOPEN_TYPE_VOID_PTR, 1, workspace);
    }
    varpack.finalize();
    return std::move(varpack);
}

} // namespace native
} // namespace at