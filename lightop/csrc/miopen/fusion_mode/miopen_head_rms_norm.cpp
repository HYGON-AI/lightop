
#include "miopen/miopen-wrapper.h"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <ATen/ATen.h>
#include "miopen/Descriptors.h"

namespace at {
namespace native {

Tensor miopen_head_rms_norm(
    const Tensor& x, const Tensor& weight, const float epsilon, const float head_offset, const float head_num)
{
    TensorDescriptor x_tensor(x);
    TensorDescriptor weight_tensor(weight);

    Tensor out_y = at::empty_like(x);

    TensorDescriptor output_y(out_y);
    MIOPEN_CHECK(miopenHeadRMSNormForward(getMiopenHandle_new(),
                                          x_tensor.desc(),
                                          x.data_ptr(),
                                          weight_tensor.desc(),
                                          weight.data_ptr(),
                                          epsilon,
                                          head_offset,
                                          head_num,
                                          output_y.desc(),
                                          out_y.data_ptr()));
    return std::move(out_y);
}

} // namespace native
} // namespace at
