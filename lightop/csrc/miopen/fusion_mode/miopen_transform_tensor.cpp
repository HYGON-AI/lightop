#include "miopen/miopen-wrapper.h"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <ATen/ATen.h>
#include "miopen/Descriptors.h"
#include <iostream>
namespace at {
namespace native {
Tensor miopen_transform_tensor(const Tensor& input,
                               const int transpose_mode = 0,
                               const float alpha        = 1.0,
                               const float beta         = 0.0)
{
    if(input.dim() != 4)
        MIOPEN_THROW("Transform_Tensor: Only Support 2D.");
    if((transpose_mode == 2 || transpose_mode == 3) && input.sizes()[1] % 32 != 0)
        MIOPEN_THROW("Transform_Tensor: Wrong channel value, must be divided by 32.");

    const int original_layout  = transpose_mode == 0 ? 0 : transpose_mode == 1 ? 1 : 10;
    const int transpose_layout = transpose_mode == 2 ? 0 : transpose_mode == 3 ? 1 : 10;

    TensorDescriptor input_tensor(input, 0, original_layout);

    const int n = input.sizes()[0];
    const int h = input.sizes()[2];
    const int w = input.sizes()[3];
    int c       = input.sizes()[1];

    if(c % 32 != 0)
        c = (c + 31) / 32 * 32;

    TensorDescriptor out_tensor(input, 0, transpose_layout);
    Tensor out = (transpose_mode == 0 || transpose_mode == 1)
                     ? at::empty_strided({n, c / 32, h, w, 32}, {c * h * w, h * w * 32, w * 32, 32, 1}, input.options())
                 : transpose_mode == 3
                     ? at::empty({n, c, h, w}, input.options().memory_format(at::MemoryFormat::ChannelsLast))
                     : at::empty({n, c, h, w}, input.options());

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