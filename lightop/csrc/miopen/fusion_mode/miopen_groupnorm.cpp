#include "miopen/miopen-wrapper.h"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <ATen/ATen.h>
#include "miopen/Descriptors.h"
#include <iostream>

namespace at {
namespace native {

std::vector<Tensor> miopen_groupnorm(const Tensor& x,
                                     const Tensor& weight,
                                     const Tensor& bias,
                                     const int num_groups,
                                     const float epsilon,
                                     const int mode)
{
    if(x.dim() > 5 || x.dim() < 3)
        MIOPEN_THROW("GroupNorm: Only Support 1d, 2d or 3d.");

    TensorDescriptor x_tensor(x);
    TensorDescriptor weight_tensor(weight);
    TensorDescriptor bias_tensor(bias);

    std::vector<int64_t> mean_rstd_len = {x.sizes()[0] * num_groups, 1, 1, 1};

    if(x.sizes() == 5)
    {
        mean_rstd_len.push_back(1);
    }

    Tensor out_y    = at::empty_like(x);
    Tensor out_mean = at::empty(mean_rstd_len, x.options().dtype(at::kFloat));
    Tensor out_rstd = at::empty(mean_rstd_len, x.options().dtype(at::kFloat));

    TensorDescriptor output_y(out_y);
    TensorDescriptor output_mean(out_mean);
    TensorDescriptor output_rstd(out_rstd);

    MIOPEN_CHECK(miopenGroupNormForward(getMiopenHandle_new(),
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

    return {std::move(out_y), std::move(out_mean), std::move(out_rstd)};
};

std::vector<Tensor> miopen_add_groupnorm(const Tensor& x1,
                                         const Tensor& x2,
                                         const Tensor& weight,
                                         const Tensor& bias,
                                         const int num_groups,
                                         const float epsilon,
                                         const int mode)
{
    if(x1.dim() > 5 || x1.dim() < 3)
        MIOPEN_THROW("GroupNorm: x1 Only Support 1d, 2d or 3d.");
    if(x2.dim() > 5 || x2.dim() < 3)
        MIOPEN_THROW("GroupNorm: x2 Only Support 1d, 2d or 3d.");

    TensorDescriptor x1_tensor(x1);
    TensorDescriptor x2_tensor(x2);
    TensorDescriptor weight_tensor(weight);
    TensorDescriptor bias_tensor(bias);

    std::vector<int64_t> mean_rstd_len = {x1.sizes()[0] * num_groups, 1, 1, 1};

    if(x1.sizes() == 5)
    {
        mean_rstd_len.push_back(1);
    }

    Tensor out_y    = at::empty_like(x1);
    Tensor out_mean = at::empty(mean_rstd_len, x1.options());
    Tensor out_rstd = at::empty(mean_rstd_len, x1.options());

    TensorDescriptor output_y(out_y);
    TensorDescriptor output_mean(out_mean);
    TensorDescriptor output_rstd(out_rstd);

    MIOPEN_CHECK(miopenAddGroupNormForward(getMiopenHandle_new(),
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

    return {std::move(out_y), std::move(out_mean), std::move(out_rstd)};
};

std::vector<Tensor> miopen_groupnorm_backward(const Tensor& x,
                                              const Tensor& dy,
                                              const Tensor& weight,
                                              const Tensor& mean,
                                              const Tensor& rstd,
                                              const int num_groups,
                                              const float epsilon,
                                              const int mode)
{
    if(x.dim() > 5 || x.dim() < 3)
        MIOPEN_THROW("GroupNorm: Only Support 1d, 2d or 3d.");

    TensorDescriptor x_tensor(x);
    TensorDescriptor dy_tensor(dy);
    TensorDescriptor weight_tensor(weight);
    TensorDescriptor mean_tensor(mean);
    TensorDescriptor rstd_tensor(rstd);

    std::vector<int64_t> dscale_dbias_len  = {1, x.sizes()[1], 1, 1};
    // std::vector<int64_t> mean_rstd_len    = {x.sizes()[0] * num_groups, 1, 1, 1};
    // std::vector<int64_t> dscale_dbias_len = {x.sizes()[0] * x.sizes()[1], 1, 1, 1};

    if(x.sizes() == 5)
    {
        // weight_bias_len.push_back(1);
        // mean_rstd_len.push_back(1);
        dscale_dbias_len.push_back(1);
    }

    // Tensor dy     = at::empty_like(x);
    // Tensor weight = at::empty(weight_bias_len, x.options());
    // Tensor mean   = at::empty(mean_rstd_len, x.options().dtype(at::kFloat));
    // Tensor rstd   = at::empty(mean_rstd_len, x.options().dtype(at::kFloat));

    Tensor out_dx     = at::empty_like(x);
    Tensor out_dscale = at::empty(dscale_dbias_len, x.options());
    Tensor out_dbias  = at::empty(dscale_dbias_len, x.options());

    TensorDescriptor output_dx(out_dx);
    TensorDescriptor output_dscale(out_dscale);
    TensorDescriptor output_dbias(out_dbias);

    MIOPEN_CHECK(miopenGroupNormBackward(getMiopenHandle_new(),
                                         static_cast<miopenNormMode_t>(mode),
                                         x_tensor.desc(),
                                         x.data_ptr(),
                                         dy_tensor.desc(),
                                         dy.data_ptr(),
                                         weight_tensor.desc(),
                                         weight.data_ptr(),
                                         output_dx.desc(),
                                         out_dx.data_ptr(),
                                         output_dscale.desc(),
                                         out_dscale.data_ptr(),
                                         output_dbias.desc(),
                                         out_dbias.data_ptr(),
                                         mean_tensor.desc(),
                                         mean.data_ptr(),
                                         rstd.data_ptr(),
                                         num_groups,
                                         epsilon));

    return {std::move(out_dx), std::move(out_dscale), std::move(out_dbias)};
}

} // namespace native

} // namespace at
