#include "miopen/miopen-wrapper.h"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <ATen/ATen.h>
#include "miopen/Descriptors.h"
#include <iostream>

namespace at
{
namespace native
{
    
std::vector<Tensor> layernorm(const Tensor &x,
    const Tensor &weight,
    const Tensor &bias,
    const float epsilon,
    const int normalized_dim,
    const int mode)
{
    TensorDescriptor x_tensor(x);
    TensorDescriptor weight_tensor(weight);
    TensorDescriptor bias_tensor(bias);

    std::vector<int64_t> outer_len;
    if(normalized_dim == 0)
        outer_len = {1};
    else
        outer_len = {x.sizes().begin(), x.sizes().end() - (x.sizes().size() - normalized_dim)};

    Tensor out_y = at::empty_like(x);
    Tensor out_mean = at::empty(outer_len, x.options());
    Tensor out_rstd = at::empty(outer_len, x.options());

    TensorDescriptor output_y(out_y);
    TensorDescriptor output_mean(out_mean);
    TensorDescriptor output_rstd(out_rstd);

    MIOPEN_CHECK(miopenLayerNormForward(
        getMiopenHandle_new(),
        static_cast<miopenNormMode_t>(mode),
        x_tensor.desc(),
        x.data_ptr(),
        weight_tensor.desc(),
        weight.data_ptr(),
        bias_tensor.desc(),
        bias.data_ptr(),
        epsilon,
        normalized_dim,
        output_y.desc(),
        out_y.data_ptr(),
        output_mean.desc(),
        out_mean.data_ptr(),
        output_rstd.desc(),
        out_rstd.data_ptr()));

    std::cout << "layernorm test ok" << std::endl;
    return {std::move(out_y), std::move(out_mean), std::move(out_rstd)};
    
};

std::vector<Tensor> skip_layernorm(const Tensor &x1,
                                    const Tensor &x2,
                                    const Tensor &weight,
                                    const Tensor &bias,
                                    const float epsilon,
                                    const int normalized_dim,
                                    const int mode)
{
    TensorDescriptor x1_tensor(x1);
    TensorDescriptor x2_tensor(x2);
    TensorDescriptor weight_tensor(weight);
    TensorDescriptor bias_tensor(bias);

    std::vector<int64_t> outer_len;
    if(normalized_dim == 0)
        outer_len = {1};
    else
        outer_len = {x1.sizes().begin(), x1.sizes().end() - (x1.sizes().size() - normalized_dim)};

    Tensor out_y = at::empty_like(x1);
    Tensor out_mean = at::empty(outer_len, x1.options());
    Tensor out_rstd = at::empty(outer_len, x1.options());
    Tensor out_skip = at::empty(outer_len, x1.options());
   
    TensorDescriptor output_y(out_y);
    TensorDescriptor output_mean(out_mean);
    TensorDescriptor output_rstd(out_rstd);
    TensorDescriptor output_skip(out_skip);
    

    MIOPEN_CHECK(miopenSkipLayerNormForward(
        getMiopenHandle_new(),
        static_cast<miopenNormMode_t>(mode),
        x1_tensor.desc(),
        x1.data_ptr(),
        x2_tensor.desc(),
        x2.data_ptr(),
        weight_tensor.desc(),
        weight.data_ptr(),
        bias_tensor.desc(),
        bias.data_ptr(),
        epsilon,
        normalized_dim,
        output_y.desc(),
        out_y.data_ptr(),
        output_skip.desc(),
        out_skip.data_ptr(),
        output_mean.desc(),
        out_mean.data_ptr(),
        output_rstd.desc(),
        out_rstd.data_ptr()));
    
    std::cout << "skiplayernorm test ok" << std::endl;
    return {std::move(out_y), std::move(out_skip), std::move(out_mean), std::move(out_rstd)};

};


std::vector<Tensor> add_layernorm(const Tensor &x1,
                                    const Tensor &x2,
                                    const Tensor &weight,
                                    const Tensor &bias,
                                    const float epsilon,
                                    const int normalized_dim,
                                    const int mode)
{
    TensorDescriptor x1_tensor(x1);
    TensorDescriptor x2_tensor(x2);
    TensorDescriptor weight_tensor(weight);
    TensorDescriptor bias_tensor(bias);

    std::vector<int64_t> outer_len;
    if(normalized_dim == 0)
        outer_len = {1};
    else
        outer_len = {x1.sizes().begin(), x1.sizes().end() - (x1.sizes().size() - normalized_dim)};

    Tensor out_y = at::empty_like(x1);
    Tensor out_mean = at::empty(outer_len, x1.options());
    Tensor out_rstd = at::empty(outer_len, x1.options());
    Tensor out_skip = at::empty(outer_len, x1.options());
   
    TensorDescriptor output_y(out_y);
    TensorDescriptor output_mean(out_mean);
    TensorDescriptor output_rstd(out_rstd);
    TensorDescriptor output_skip(out_skip);
    
    MIOPEN_CHECK(miopenAddLayerNormForward(
        getMiopenHandle_new(),
        static_cast<miopenNormMode_t>(mode),
        x1_tensor.desc(),
        x1.data_ptr(),
        x2_tensor.desc(),
        x2.data_ptr(),
        weight_tensor.desc(),
        weight.data_ptr(),
        bias_tensor.desc(),
        bias.data_ptr(),
        epsilon,
        normalized_dim,
        output_y.desc(),
        out_y.data_ptr(),
        output_mean.desc(),
        out_mean.data_ptr(),
        output_rstd.desc(),
        out_rstd.data_ptr()));
    
    std::cout << "addlayernorm test ok" << std::endl;
    return {std::move(out_y), std::move(out_mean), std::move(out_rstd)};


};

} // namespace native
    
} // namespace at
