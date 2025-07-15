
#include "miopen/miopen-wrapper.h"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <ATen/ATen.h>
#include "miopen/Descriptors.h"
#include <iostream>

namespace at {
namespace native {

std::vector<Tensor> add_rms_norm_dynamic_quant(
    const Tensor& x, const Tensor& x2, const Tensor& weight, const Tensor& smoothScale, const float epsilon)
{

    TensorDescriptor x_tensor(x);
    TensorDescriptor x2_tensor(x2);
    TensorDescriptor weight_tensor(weight);
    TensorDescriptor smoothScale_tensor(smoothScale);

    Tensor out_add               = at::empty_like(x);
    Tensor out_y                 = at::empty_like(x);
    Tensor out_perTokenScaleDesc = at::empty(x.sizes()[0], x.options());

    TensorDescriptor output_add(out_add);
    TensorDescriptor output_y(out_y);
    TensorDescriptor output_perTokenScaleDesc(out_perTokenScaleDesc);
    MIOPEN_CHECK(miopenAddRMSNormDynamicQuant(getMiopenHandle_new(),
                                              MIOPEN_ELEMENTWISE_AFFINE,
                                              x_tensor.desc(),
                                              x.data_ptr(),
                                              x2_tensor.desc(),
                                              x2.data_ptr(),
                                              weight_tensor.desc(),
                                              weight.data_ptr(),
                                              smoothScale_tensor.desc(),
                                              smoothScale.data_ptr(),
                                              epsilon,
                                              output_add.desc(),
                                              out_add.data_ptr(),
                                              output_y.desc(),
                                              out_y.data_ptr(),
                                              output_perTokenScaleDesc.desc(),
                                              out_perTokenScaleDesc.data_ptr()));

    std::cout << "add_rms_norm_dynamic_quant test ok" << std::endl;
    return {std::move(out_add), std::move(out_y), std::move(out_perTokenScaleDesc)};
}

} // namespace native
} // namespace at
