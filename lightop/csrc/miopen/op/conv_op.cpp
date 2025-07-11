#include "miopen/miopen-wrapper.h"
#include "miopen/graph/tensor.hpp"
#include <stdint.h>
#include <vector>
namespace at {
namespace native {
BackendDescriptor CreatConvOp(miopenDataType_t data_type,
                              BackendDescriptor input,
                              BackendDescriptor weight,
                              std::vector<int64_t> pads,
                              std::vector<int64_t> strides,
                              std::vector<int64_t> dilations,
                              float alpha,
                              float beta,
                              BackendDescriptor output)
{

    BackendDescriptor conv_desc(MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR);
    miopenConvolutionMode_t mode = miopenConvolution;
    const size_t spatial_dims    = pads.size();

    conv_desc.set(MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS, MIOPEN_TYPE_INT64, 1, &spatial_dims);
    conv_desc.set(MIOPEN_ATTR_CONVOLUTION_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &data_type);
    conv_desc.set(MIOPEN_ATTR_CONVOLUTION_CONV_MODE, MIOPEN_TYPE_CONVOLUTION_MODE, 1, &mode);
    conv_desc.set(
        MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS, MIOPEN_TYPE_INT64, pads.size(), pads.data());
    conv_desc.set(
        MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS, MIOPEN_TYPE_INT64, pads.size(), pads.data());
    conv_desc.set(
        MIOPEN_ATTR_CONVOLUTION_DILATIONS, MIOPEN_TYPE_INT64, dilations.size(), dilations.data());
    conv_desc.set(
        MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES, MIOPEN_TYPE_INT64, strides.size(), strides.data());
    conv_desc.finalize();

    BackendDescriptor conv_operator(MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    conv_operator.set(MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                      MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                      1,
                      conv_desc.desc());
    conv_operator.set(MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                      MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                      1,
                      input.desc());
    conv_operator.set(MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                      MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                      1,
                      weight.desc());
    conv_operator.set(MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                      MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                      1,
                      output.desc());

    conv_operator.set(
        MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, MIOPEN_TYPE_FLOAT, 1, &alpha);
    conv_operator.set(MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, MIOPEN_TYPE_FLOAT, 1, &beta);

    conv_operator.finalize();
    return conv_operator;
}
} // namespace native
} // namespace at