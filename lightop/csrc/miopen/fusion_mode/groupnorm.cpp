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
    
std::vector<Tensor> groupnorm(const Tensor &x,
    const Tensor &weight,
    const Tensor &bias,
    const int num_groups,
    const float epsilon,
    const int mode,
    const std::string &layout)
{
    TensorDescriptor x_tensor(x, layout);
    TensorDescriptor weight_tensor(weight, layout);
    TensorDescriptor bias_tensor(bias, layout);

    std::vector<int64_t> mean_rstd_len   = {x.sizes()[0] * num_groups, 1, 1, 1};

    if(x.sizes() == 5)
    {
        mean_rstd_len.push_back(1);
    }

    Tensor out_y = at::empty_like(x);
    Tensor out_mean = at::empty(mean_rstd_len, x.options());
    Tensor out_rstd = at::empty(mean_rstd_len, x.options());

    TensorDescriptor output_y(out_y, layout);
    TensorDescriptor output_mean(out_mean, layout);
    TensorDescriptor output_rstd(out_rstd, layout);

    MIOPEN_CHECK(miopenGroupNormForward(
        getMiopenHandle_new(),
        static_cast<miopenNormMode_t>(mode),
        x_tensor.desc(),
        x.data_ptr(),
        weight_tensor.desc(),
        weight.data_ptr(),
        bias_tensor.desc(),
        bias.data_ptr(),
        num_groups,
        epsilon,
        output_y.desc(),
        out_y.data_ptr(),
        output_mean.desc(),
        out_mean.data_ptr(),
        output_rstd.desc(),
        out_rstd.data_ptr()));

    std::cout << "groupnorm test ok" << std::endl;
    return {std::move(out_y), std::move(out_mean), std::move(out_rstd)};

};

std::vector<Tensor> add_groupnorm(const Tensor &x1,
    const Tensor &x2,
    const Tensor &weight,
    const Tensor &bias,
    const int num_groups,
    const float epsilon,
    const int mode,
    const std::string &layout)
{
    TensorDescriptor x1_tensor(x1, layout);
    TensorDescriptor x2_tensor(x2, layout);
    TensorDescriptor weight_tensor(weight, layout);
    TensorDescriptor bias_tensor(bias, layout);

    std::vector<int64_t> mean_rstd_len   = {x1.sizes()[0] * num_groups, 1, 1, 1};

    if(x1.sizes() == 5)
    {
        mean_rstd_len.push_back(1);
    }

    Tensor out_y = at::empty_like(x1);
    Tensor out_mean = at::empty(mean_rstd_len, x1.options());
    Tensor out_rstd = at::empty(mean_rstd_len, x1.options());

    TensorDescriptor output_y(out_y, layout);
    TensorDescriptor output_mean(out_mean, layout);
    TensorDescriptor output_rstd(out_rstd, layout);

    MIOPEN_CHECK(miopenAddGroupNormForward(
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
        num_groups,
        epsilon,
        output_y.desc(),
        out_y.data_ptr(),
        output_mean.desc(),
        out_mean.data_ptr(),
        output_rstd.desc(),
        out_rstd.data_ptr()));

    std::cout << "addgroupnorm test ok" << std::endl;
    return {std::move(out_y), std::move(out_mean), std::move(out_rstd)};

};

} // namespace native
    
} // namespace at
