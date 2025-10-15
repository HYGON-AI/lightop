#include "miopen/miopen-wrapper.h"
#include "miopen/graph/tensor.hpp"
#include <stdint.h>
#include <vector>
namespace at {
namespace native {
std::pair<BackendDescriptor, BackendDescriptor> CreatGemmOp(miopenDataType_t data_type,
                                                            const int batch_size,
                                                            const BackendDescriptor& matmul_input_a,
                                                            const BackendDescriptor& matmul_input_b,
                                                            const BackendDescriptor& matmul_output)
{
    BackendDescriptor gemm_desc(MIOPEN_BACKEND_MATMUL_DESCRIPTOR);
    gemm_desc.set(MIOPEN_ATTR_MATMUL_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &data_type);
    gemm_desc.finalize();

    BackendDescriptor gemm_operator(MIOPEN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
    gemm_operator.set(
        MIOPEN_ATTR_OPERATION_MATMUL_ADESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, matmul_input_a.storage_addr());
    gemm_operator.set(
        MIOPEN_ATTR_OPERATION_MATMUL_BDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, matmul_input_b.storage_addr());
    gemm_operator.set(
        MIOPEN_ATTR_OPERATION_MATMUL_CDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, matmul_output.storage_addr());
    gemm_operator.set(MIOPEN_ATTR_OPERATION_MATMUL_DESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, gemm_desc.storage_addr());
    gemm_operator.set(MIOPEN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT, MIOPEN_TYPE_INT64, 1, &batch_size);

    gemm_operator.finalize();
    return std::make_pair(std::move(gemm_operator), std::move(gemm_desc));
}
} // namespace native

} // namespace at
