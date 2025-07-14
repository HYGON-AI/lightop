#include "miopen/Descriptors.h"
#include "miopen/graph/tensor.hpp"

namespace at {
namespace native {

BackendDescriptor CreatPointWiseOp(miopenDataType_t data_type,
                                   miopenPointwiseMode_t mode,
                                   const BackendDescriptor& pre_operator_output,
                                   const BackendDescriptor& input,
                                   const BackendDescriptor& output)
{

    BackendDescriptor point_wise_desc(MIOPEN_BACKEND_POINTWISE_DESCRIPTOR);
    point_wise_desc.set(MIOPEN_ATTR_POINTWISE_MODE, MIOPEN_TYPE_POINTWISE_MODE, 1, &mode);
    point_wise_desc.set(MIOPEN_ATTR_POINTWISE_MATH_PREC, MIOPEN_TYPE_DATA_TYPE, 1, &data_type);
    point_wise_desc.finalize();

    BackendDescriptor operation(MIOPEN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    operation.set(MIOPEN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                  MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                  1,
                  point_wise_desc.storage_addr());
    operation.set(
        MIOPEN_ATTR_OPERATION_POINTWISE_XDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, pre_operator_output.storage_addr());
    operation.set(MIOPEN_ATTR_OPERATION_POINTWISE_BDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, input.storage_addr());
    operation.set(MIOPEN_ATTR_OPERATION_POINTWISE_YDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, output.storage_addr());

    operation.finalize();
    return std::move(operation);
}
BackendDescriptor CreatPointWiseOp(miopenDataType_t data_type,
                                   miopenPointwiseMode_t mode,
                                   const BackendDescriptor& pre_operator_output,
                                   const BackendDescriptor& output)
{

    BackendDescriptor point_wise_desc(MIOPEN_BACKEND_POINTWISE_DESCRIPTOR);
    point_wise_desc.set(MIOPEN_ATTR_POINTWISE_MODE, MIOPEN_TYPE_POINTWISE_MODE, 1, &mode);
    point_wise_desc.set(MIOPEN_ATTR_POINTWISE_MATH_PREC, MIOPEN_TYPE_DATA_TYPE, 1, &data_type);
    point_wise_desc.finalize();

    BackendDescriptor operation(MIOPEN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    operation.set(MIOPEN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                  MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                  1,
                  point_wise_desc.storage_addr());
    operation.set(
        MIOPEN_ATTR_OPERATION_POINTWISE_XDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, pre_operator_output.storage_addr());
    operation.set(MIOPEN_ATTR_OPERATION_POINTWISE_YDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, output.storage_addr());

    operation.finalize();
    return std::move(operation);
}
} // namespace native
} // namespace at