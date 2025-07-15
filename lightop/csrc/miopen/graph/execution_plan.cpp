
#include "miopen/graph/execution_plan.hpp"
#include "miopen/Handle.h"
#include <vector>
namespace at {
namespace native {
// exec plan
BackendDescriptor CreatExeutionPlan(const std::vector<miopenBackendDescriptor_t>& ops_operation)

{
    auto handle = getMiopenHandle_new();
    // graph
    BackendDescriptor fusion_op_graph(MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    fusion_op_graph.set(
        MIOPEN_ATTR_OPERATIONGRAPH_OPS, MIOPEN_TYPE_BACKEND_DESCRIPTOR, ops_operation.size(), ops_operation.data());
    fusion_op_graph.set(MIOPEN_ATTR_OPERATIONGRAPH_HANDLE, MIOPEN_TYPE_HANDLE, 1, &handle);
    fusion_op_graph.finalize();

    // engine
    BackendDescriptor engine(MIOPEN_BACKEND_ENGINE_DESCRIPTOR);

    engine.set(MIOPEN_ATTR_ENGINE_OPERATION_GRAPH, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, fusion_op_graph.storage_addr());
    // TODO: check  gidx
    int64_t gidx = 0;
    engine.set(MIOPEN_ATTR_ENGINE_GLOBAL_INDEX, MIOPEN_TYPE_INT64, 1, &gidx);
    engine.finalize();
    // engcfg
    BackendDescriptor engcfg(MIOPEN_BACKEND_ENGINECFG_DESCRIPTOR);
    engcfg.set(MIOPEN_ATTR_ENGINECFG_ENGINE, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, engine.storage_addr());
    engcfg.finalize();

    BackendDescriptor exec_plan(MIOPEN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);

    exec_plan.set(MIOPEN_ATTR_EXECUTION_PLAN_HANDLE, MIOPEN_TYPE_HANDLE, 1, &handle);

    exec_plan.set(MIOPEN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, engcfg.storage_addr());
    exec_plan.finalize();
    return exec_plan;
}
} // namespace native
} // namespace at