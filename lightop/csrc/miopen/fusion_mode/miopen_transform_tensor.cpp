#include "miopen/miopen-wrapper.h"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <ATen/ATen.h>
#include "miopen/Descriptors.h"
#include <iostream>
namespace at {
namespace native {

enum class TRANSFROM_MODE
{
    NCHWToNCHWc32 = 0,
    NHWCToNCHWc32 = 1,
    NCHWc32ToNCHW = 2,
    NCHWc32ToNHWC = 3
};

Tensor miopen_transform_tensor(const Tensor& input,
                               const int transpose_mode = 0,
                               const float alpha        = 1.0,
                               const float beta         = 0.0)
{
    if(input.dim() != 4)
        MIOPEN_THROW("Transform_Tensor: Only Support 2D.");

    TRANSFROM_MODE mode = static_cast<TRANSFROM_MODE>(transpose_mode);
    if((mode == TRANSFROM_MODE::NCHWc32ToNCHW || mode == TRANSFROM_MODE::NCHWc32ToNHWC) && input.sizes()[1] % 32 != 0)
        MIOPEN_THROW("Transform_Tensor: Wrong channel value, must be divided by 32.");

    const miopenTensorLayout_t original_layout  = mode == TRANSFROM_MODE::NCHWToNCHWc32   ? miopenTensorNCHW
                                                  : mode == TRANSFROM_MODE::NHWCToNCHWc32 ? miopenTensorNHWC
                                                                                          : miopenTensorNCHWc32;
    const miopenTensorLayout_t transpose_layout = mode == TRANSFROM_MODE::NCHWc32ToNCHW   ? miopenTensorNCHW
                                                  : mode == TRANSFROM_MODE::NCHWc32ToNHWC ? miopenTensorNHWC
                                                                                          : miopenTensorNCHWc32;

    TensorDescriptor input_tensor(input, original_layout);

    const int n = input.sizes()[0];
    const int h = input.sizes()[2];
    const int w = input.sizes()[3];
    int c       = input.sizes()[1];

    if(c % 32 != 0)
        c = (c + 31) / 32 * 32;

    TensorDescriptor out_tensor(input, transpose_layout);
    Tensor out = at::empty_strided({n, c / 32, h, w, 32}, {c * h * w, h * w * 32, w * 32, 32, 1}, input.options());

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
