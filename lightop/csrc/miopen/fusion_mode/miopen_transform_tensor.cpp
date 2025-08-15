#include "miopen/miopen-wrapper.h"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <ATen/ATen.h>
#include "miopen/Descriptors.h"
#include <iostream>
namespace at {
namespace native {
Tensor miopen_transform_tensor(const Tensor& input, const float alpha = 1.0, const float beta = 0.0)
{
    if(!(input.dim() == 4 || input.dim() == 5))
        MIOPEN_THROW("Transform_Tensor: Only Support 2D or 3D.");
    
    TensorDescriptor input_tensor(input);
    Tensor out = at::empty_like(input);
    TensorDescriptor out_tensor(out, 0, 10);

    MIOPEN_CHECK(miopenTransformTensor(getMiopenHandle_new(),
                                       &alpha,
                                       input_tensor.desc(),
                                       input.data_ptr(),
                                       &beta,
                                       out_tensor.desc(),
                                       out.data_ptr()));

    return out;
}
} // namespace native
} // namespace at