#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Resize.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <fuse_bn_relu.cuh>
#include <fuse_bn_relu_miopen.cuh>
#include <torch/extension.h>
namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor, Tensor> bnrelu_backward_reduce_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& output, const Tensor& mean, const Tensor& invstd, const c10::optional<Tensor>& weight_opt, bool input_g, bool weight_g, bool bias_g);
std::tuple<Tensor, Tensor> bnrelu_gather_stats_with_counts_cuda(
    const Tensor& self, const Tensor& mean, const Tensor& invstd, const c10::optional<Tensor>& running_mean_opt /* optional */, const c10::optional<Tensor>& running_var_opt /* optional */, double momentum, double epsilon, const Tensor& counts);

namespace {

ScalarType first_type() {
  return ScalarType::Undefined;
}

template <typename... Args>
ScalarType first_type(const Tensor& arg, const Args&... parameters) {
  return arg.defined() ? arg.scalar_type() : first_type(parameters...);
}

// A transform is mixed type if the parameters are higher precision than the input
template <typename... Args>
bool is_mixed_type(const Tensor& input, const Args&... parameters) {
  const auto parameter_type = first_type(parameters...);
  return ((parameter_type != ScalarType::Undefined) &&
          (parameter_type != input.scalar_type()));
}

inline bool bnrelu_use_channels_last_kernels(const at::Tensor& self) {
  return (self.is_contiguous(at::MemoryFormat::ChannelsLast) ||
          (self.is_contiguous() && self.strides()[1] == 1));
}

enum class Impl {
  Contiguous,
  ChannelsLast,
  General,
};

inline Impl bnrelu_choose_impl(const Tensor& self) {
  if (!at::cuda::detail::canUse32BitIndexMath(self)) {
    return Impl::General;
  }

  if (self.is_contiguous()) {
    return self.strides()[1] == 1 ? Impl::ChannelsLast : Impl::Contiguous;
  }

  if (self.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    return Impl::ChannelsLast;
  }

  return Impl::General;
}

inline Impl bnrelu_choose_impl(const Tensor& in1, const Tensor& in2) {
  auto imp1 = bnrelu_choose_impl(in1);
  if (imp1 == Impl::General) {
    return imp1;
  }
  auto imp2 = bnrelu_choose_impl(in2);
  return imp1 == imp2 ? imp1 : Impl::General;
}

void bnrelu_elementwise(
    const Tensor& out, const Tensor& out_relu, const Tensor& self, const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt, const Tensor& mean_, const Tensor& invstd_, int fuse_act_mode) {
  switch (bnrelu_choose_impl(self)) {
  case Impl::Contiguous: {
    c10::MaybeOwned<Tensor> weight = at::borrow_from_optional_tensor(weight_opt);
    c10::MaybeOwned<Tensor> bias = at::borrow_from_optional_tensor(bias_opt);
    resize_output(out, self.sizes());
    resize_output(out_relu, self.sizes());
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(),
                                    "bnrelu_elementwise_cuda", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      const bool mixed_type = is_mixed_type(self, *weight, *bias);
      if (mixed_type) {
        bnrelu_elemt_cuda_template<scalar_t, accscalar_t, int32_t>(
            out, out_relu, self, *weight, *bias, mean_, invstd_, fuse_act_mode);
      } else {
        bnrelu_elemt_cuda_template<scalar_t, scalar_t, int32_t>(
            out, out_relu, self, *weight, *bias, mean_, invstd_, fuse_act_mode);
      }
    });
    return;
  }
  case Impl::ChannelsLast: {
    auto weight = at::borrow_from_optional_tensor(weight_opt);
    auto bias = at::borrow_from_optional_tensor(bias_opt);

    if (resize_output_check(out, self.sizes())) {
        resize_impl_cuda_(out.unsafeGetTensorImpl(), self.sizes(), self.strides());
    }
    if ((out.strides() == self.strides()) &&
        (!weight->defined() || weight->is_contiguous()) &&
        (!bias->defined() || bias->is_contiguous()) &&
        (!mean_.defined() || mean_.is_contiguous()) &&
        (!invstd_.defined() || invstd_.is_contiguous())) {
      bnrelu_elemt_channels_last_cuda_template(
          out, out_relu, self, *weight, *bias, mean_, invstd_);
      return;
    }
    C10_FALLTHROUGH;
  }
  case Impl::General: {
    const int64_t ndim = self.dim();
    DimVector sizes(ndim, 1), strides(ndim, 0);
    // Helper to convert 1d tensors to an nd tensor that broadcasts with input
    // All elements go into the channel dimension
    auto as_nd = [&](const Tensor& t) {
      TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
      sizes[1] = t.sizes()[0];
      strides[1] = t.strides()[0];
      return t.as_strided(sizes, strides);
    };

    auto weight = weight_opt.has_value() && weight_opt->defined() ?
        as_nd(*weight_opt) : at::scalar_tensor(1, mean_.options());
    auto bias = bias_opt.has_value() && bias_opt->defined() ?
        as_nd(*bias_opt) : at::scalar_tensor(0, mean_.options());
    auto mean = as_nd(mean_);
    auto invstd = as_nd(invstd_);

    auto iter = TensorIteratorConfig()
        .add_output(out)
        .add_input(self)
        .add_input(weight)
        .add_input(bias)
        .add_input(mean)
        .add_input(invstd)
        .check_all_same_dtype(false)
        .promote_inputs_to_common_dtype(false)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(),
                                    "bnrelu_elementwise_cuda", [&] {
      using acc_t = at::acc_type<scalar_t, true>;
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t input, acc_t weight, acc_t bias,
                                      acc_t mean, acc_t invstd) -> scalar_t {
        return ((input - mean) * invstd) * weight + bias;
      });
    });
    return;
  }
  }
}

Tensor bnrelu_elementwise_backward_train(
    const Tensor& grad_out, const Tensor& input, const Tensor& output, const Tensor& mean, const Tensor& invstd,
    const Tensor& weight, const Tensor& sum_dy, const Tensor& sum_dy_xmu) {
  switch (bnrelu_choose_impl(input, grad_out)) {
  case Impl::Contiguous: {
    return AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                           "bnrelu_backward_elemt", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      const bool mixed_type = is_mixed_type(input, weight);
      if (mixed_type) {
        return bnrelu_backward_elemt_cuda_template<scalar_t, accscalar_t, int32_t>(
            grad_out, input, output, mean, invstd, weight, sum_dy, sum_dy_xmu);
      } else {
        return bnrelu_backward_elemt_cuda_template<scalar_t, scalar_t, int32_t>(
            grad_out, input, output, mean, invstd, weight, sum_dy, sum_dy_xmu);
      }
    });
  }
  case Impl::ChannelsLast: {
    if ((!weight.defined() || weight.is_contiguous()) &&
        mean.is_contiguous() && invstd.is_contiguous()) {
      return bnrelu_backward_elemt_channels_last_cuda_template(
          grad_out, input, output, mean, invstd, weight, sum_dy, sum_dy_xmu);
    }
    C10_FALLTHROUGH;
  }
  case Impl::General: {
    const auto ndim = input.dim();
    DimVector sizes(ndim, 1), strides(ndim, 0);
    auto as_nd = [&](const Tensor& t) {
      TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
      sizes[1] = t.sizes()[0];
      strides[1] = t.strides()[0];
      return t.as_strided(sizes, strides);
    };
    auto invstd_nd = as_nd(invstd);
    auto mean_nd = as_nd(mean);
    auto sum_dy_nd = as_nd(sum_dy);
    auto sum_dy_xmu_nd = as_nd(sum_dy_xmu);
    auto weight_nd = weight.defined() ? as_nd(weight) :
        at::scalar_tensor(1.0, input.options().dtype(mean.scalar_type()));

    Tensor grad_input = at::empty(input.sizes(), grad_out.options());
    auto iter = TensorIteratorConfig()
        .add_output(grad_input)
        .add_input(grad_out)
        .add_input(input)
        .add_input(weight_nd)
        .add_input(mean_nd)
        .add_input(invstd_nd)
        .add_input(sum_dy_xmu_nd)
        .add_input(sum_dy_nd)
        .check_all_same_dtype(false)
        .promote_inputs_to_common_dtype(false)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, grad_out.scalar_type(),
                                    "bnrelu_eval_backward", [&]{
      using accscalar_t = at::acc_type<scalar_t, true>;
      auto norm_fct = static_cast<accscalar_t>(1.0 / (input.numel() /input.size(1)) );
      gpu_kernel(iter, [norm_fct] GPU_LAMBDA (scalar_t gO, scalar_t input, accscalar_t weight,
                                              accscalar_t mean, accscalar_t invstd,
                                              accscalar_t xmu, accscalar_t dy) -> scalar_t {
        auto factor_1_c = invstd * invstd * xmu * norm_fct;
        auto factor_2_c = weight * invstd;
        auto m_dy_c = dy * norm_fct;
        return (gO - m_dy_c - (input - mean) * factor_1_c) * factor_2_c;
      });
    });
    return grad_input;
  }
  }
  TORCH_INTERNAL_ASSERT(false);
}

Tensor bnrelu_elementwise_backward_eval(
    const Tensor& grad_out, const Tensor& input, const Tensor& output,
    const Tensor& invstd, const Tensor& weight) {
  const auto ndim = input.dim();
  DimVector shape(ndim, 1), strides(ndim, 0);
  shape[1] = invstd.sizes()[0];
  strides[1] = invstd.strides()[0];
  auto invstd_nd = invstd.as_strided(shape, strides);
  Tensor grad_input = at::empty(input.sizes(), grad_out.options());

  if (weight.defined()) {
    strides[1] = weight.strides()[0];
    auto weight_nd = weight.as_strided(shape, strides);
    auto iter = TensorIteratorConfig()
        .add_output(grad_input)
        .add_input(grad_out)
        .add_input(invstd_nd)
        .add_input(weight_nd)
        .check_all_same_dtype(false)
        .promote_inputs_to_common_dtype(false)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, grad_out.scalar_type(),
                                    "bnrelu_eval_backward", [&]{
      using accscalar_t = at::acc_type<scalar_t, true>;
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t gO, accscalar_t invstd, accscalar_t weight)
                 -> scalar_t {
          return gO * weight * invstd;
      });
    });
  } else {
    auto iter = TensorIteratorConfig()
        .add_output(grad_input)
        .add_input(grad_out)
        .add_input(invstd_nd)
        .check_all_same_dtype(false)
        .promote_inputs_to_common_dtype(false)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, grad_out.scalar_type(),
                                    "bnrelu_eval_backward", [&]{
      using accscalar_t = at::acc_type<scalar_t, true>;
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t gO, accscalar_t invstd) -> scalar_t {
          return gO * invstd;
      });
    });
  }
  return grad_input;
}


void bnrelu_mean_var(const Tensor& self, Tensor& save_mean, Tensor& save_var) {
  // NOTE: Epsilon is only used for InvStd, not Var. The value here is ignored.
  const double dummy_epsilon = 1e-5;
  switch (bnrelu_choose_impl(self)) {
  case Impl::Contiguous: {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, self.scalar_type(), "bnrelu_stats_cuda", [&] {
      bnrelu_stats_cuda_template<scalar_t, int32_t, Var>(
          save_mean, save_var, self, dummy_epsilon);
    });
    return;
  }
  case Impl::ChannelsLast: {
    if ((!save_mean.defined() || save_mean.is_contiguous()) &&
        (!save_var.defined() || save_var.is_contiguous())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf, kBFloat16, self.scalar_type(), "bnrelu_stats_cuda", [&] {
        bnrelu_stats_channels_last_cuda_template<scalar_t, Var>(
            save_mean, save_var, self, dummy_epsilon);
      });
      return;
    }
    C10_FALLTHROUGH;
  }
  case Impl::General: {
    const int64_t ndim = self.dim();
    DimVector reduce_dims(ndim - 1);
    reduce_dims[0] = 0;
    for (int64_t i = 2; i < ndim; ++i) {
      reduce_dims[i - 1] = i;
    }

    // For some reason this isn't an actual operator but it exists anyway...
    at::native::var_mean_out(save_var, save_mean, self, /*dims=*/reduce_dims,
                            /*unbiased=*/false, /*keepdim=*/false);
    return;
  }
  }
}

void bnrelu_update_stats(
    const Tensor& save_mean, const Tensor& save_var,
    const Tensor& running_mean, const Tensor& running_var,
    double momentum_, int64_t N) {

  auto iter = TensorIteratorConfig()
      .add_output(running_mean)
      .add_output(running_var)
      .add_input(save_mean)
      .add_input(save_var)
      .add_input(running_mean)
      .add_input(running_var)
      .check_all_same_dtype(false)
      .promote_inputs_to_common_dtype(false)
      .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, running_mean.scalar_type(),
                                  "bnrelu_update_stats_cuda", [&] {
      using acc_t = at::acc_type<scalar_t, true>;
      const auto bessel_correction_factor = static_cast<acc_t>(
          static_cast<double>(N) / static_cast<double>(N - 1));
      const auto momentum = static_cast<acc_t>(momentum_);
      gpu_kernel_multiple_outputs(
          iter, [=] GPU_LAMBDA (acc_t mean, acc_t var, scalar_t running_mean, scalar_t running_var)
               -> thrust::tuple<scalar_t, scalar_t> {
        const auto unbiased_var = var * bessel_correction_factor;
        return thrust::tuple<scalar_t, scalar_t>{
          mean * momentum + (1 - momentum) * running_mean,
          unbiased_var * momentum + (1 - momentum) * running_var,
        };
      });
  });
}

void bnrelu_update_stats_and_invert(
    const Tensor& save_mean, const Tensor& save_var,
    const Tensor& running_mean, const Tensor& running_var,
    double momentum_, double epsilon, int64_t N) {

  auto iter = TensorIteratorConfig()
      .add_output(running_mean)
      .add_output(running_var)
      .add_output(save_var)
      .add_input(save_mean)
      .add_input(save_var)
      .add_input(running_mean)
      .add_input(running_var)
      .check_all_same_dtype(false)
      .promote_inputs_to_common_dtype(false)
      .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, running_mean.scalar_type(),
                                  "bnrelu_update_stats_cuda", [&] {
      using acc_t = at::acc_type<scalar_t, true>;
      const auto bessel_correction_factor = static_cast<acc_t>(
          static_cast<double>(N) / static_cast<double>(N - 1));
      const auto eps = static_cast<acc_t>(epsilon);
      const auto momentum = static_cast<acc_t>(momentum_);
      gpu_kernel_multiple_outputs(
          iter, [=] GPU_LAMBDA (acc_t mean, acc_t var, scalar_t running_mean, scalar_t running_var)
               -> thrust::tuple<scalar_t, scalar_t, acc_t> {
        const auto unbiased_var = var * bessel_correction_factor;
        return thrust::tuple<scalar_t, scalar_t, acc_t>{
          mean * momentum + (1 - momentum) * running_mean,
          unbiased_var * momentum + (1 - momentum) * running_var,
          c10::cuda::compat::rsqrt(var + eps)
        };
      });
  });
}

void bnrelu_calc_invstd(const Tensor& out_invstd, const Tensor& running_var, double epsilon) {
  auto iter = TensorIteratorConfig()
      .add_output(out_invstd)
      .add_input(running_var)
      .check_all_same_dtype(false)
      .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, running_var.scalar_type(),
                                  "bnrelu_invert_std_cuda", [&] {
    using acc_t = at::acc_type<scalar_t, true>;
    auto eps = static_cast<acc_t>(epsilon);
    gpu_kernel(iter, [eps] GPU_LAMBDA (scalar_t var) -> acc_t {
      return c10::cuda::compat::rsqrt(var + eps);
    });
  });
}
}

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> bnrelu_cuda_out(const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, bool train, double momentum, double epsilon, Tensor& output, Tensor& output_relu, Tensor& save_mean, Tensor& save_invstd, int fuse_act_mode) {
  const bool has_running_mean = (running_mean_opt.has_value() && running_mean_opt->defined());
  const bool has_running_var = (running_var_opt.has_value() && running_var_opt->defined());
  TORCH_CHECK(has_running_mean == has_running_var);

  if (train) {
    bnrelu_mean_var(self, save_mean, save_invstd);
    if (has_running_mean) {
      const int64_t N = self.numel() / save_mean.numel();
      bnrelu_update_stats_and_invert(
          save_mean, save_invstd, *running_mean_opt, *running_var_opt,
          momentum, epsilon, N);
    } else {
      bnrelu_calc_invstd(save_invstd, save_invstd, epsilon);
    }
  } else {
    TORCH_CHECK(has_running_mean);
    at::native::resize_output(save_mean, running_mean_opt->sizes());
    save_mean.copy_(*running_mean_opt, /*non_blocking=*/true);
    bnrelu_calc_invstd(save_invstd, running_var_opt.value(), epsilon);
  }

  bnrelu_elementwise(output, output_relu, self, weight_opt, bias_opt, save_mean, save_invstd, fuse_act_mode);
  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>(output, output_relu, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> bnrelu_cuda(const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, bool train, double momentum, double epsilon, int fuse_act_mode) {
  auto output = at::empty_like(self);
  int64_t n_input = self.size(1);
  auto options = self.options().dtype(
      at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true));
  auto save_mean = at::empty({n_input}, options);
  auto save_invstd = at::empty({n_input}, options);
  Tensor output_relu;
  if (fuse_act_mode == fuse_act_relu) {
    output_relu = at::empty_like(self, self.options().dtype(kBool));
  } else {
    output_relu = at::empty_like(self);
  }

  at::native::bnrelu_cuda_out(
      self,
      weight_opt,
      bias_opt,
      running_mean_opt,
      running_var_opt,
      train,
      momentum,
      epsilon,
      output,
      output_relu,
      save_mean,
      save_invstd,
      fuse_act_mode);
  return std::make_tuple(output, output_relu, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> bnrelu_backward_cuda(const Tensor& grad_out, const Tensor& input, const Tensor& output, 
       const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, 
       const c10::optional<Tensor>& save_mean_opt, const c10::optional<Tensor>& save_invstd_opt, bool train, double epsilon, 
       std::array<bool,3> grad_input_mask, int fuse_act_mode) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight = at::borrow_from_optional_tensor(weight_opt);
  c10::MaybeOwned<Tensor> save_mean = at::borrow_from_optional_tensor(save_mean_opt);
  c10::MaybeOwned<Tensor> save_invstd = at::borrow_from_optional_tensor(save_invstd_opt);
  c10::MaybeOwned<Tensor> running_mean = at::borrow_from_optional_tensor(running_mean_opt);
  c10::MaybeOwned<Tensor> running_var = at::borrow_from_optional_tensor(running_var_opt);

  const bool needs_reduction = train || grad_input_mask[1] || grad_input_mask[2];

  // Fused reducion & elementwise kernel
  if (needs_reduction && grad_input_mask[0] &&
      !bnrelu_use_channels_last_kernels(input) &&
      cuda::detail::canUse32BitIndexMath(input) &&
      cuda::detail::canUse32BitIndexMath(grad_out)) {
    return AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                           "bnrelu_backward_cuda", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      const bool mixed_type = is_mixed_type(input, *weight, *running_mean, *running_var);
      if (mixed_type) {
          return bnrelu_backward_cuda_template<scalar_t, accscalar_t, int32_t>(
              grad_out, input, output, *weight, *running_mean, *running_var,
              *save_mean, *save_invstd, train, epsilon, grad_input_mask, fuse_act_mode);
      } else {
          return bnrelu_backward_cuda_template<scalar_t, scalar_t, int32_t>(
              grad_out, input, output, *weight, *running_mean, *running_var,
              *save_mean, *save_invstd, train, epsilon, grad_input_mask, fuse_act_mode);
      }
    });
  }

  // NOTE: native_bnrelu always returns save_mean and save_invstd to be reused in backward.
  // However, this is also called from cudnn_bnrelu in eval mode which doesn't give
  // save_mean and save_invstd, so it needs recalculated.
  const auto acc_type = at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
  Tensor mean;
  TORCH_INTERNAL_ASSERT(save_mean->defined(), "save_mean should always be defined\n");
  if (save_mean->numel() != 0) {
    mean = *save_mean;
  } else if (needs_reduction) {
    TORCH_CHECK(!train && running_mean->defined());
    mean = (running_mean->scalar_type() == acc_type) ?
        *running_mean : running_mean->to(acc_type);
  }

  Tensor invstd;
  TORCH_INTERNAL_ASSERT(save_invstd->defined(), "save_invstd should always be defined\n");
  if (save_invstd->numel() != 0) {
    invstd = *save_invstd;
  } else {
    TORCH_CHECK(!train && running_var->defined());
    auto n_channels = input.sizes()[1];
    invstd = at::empty({n_channels}, input.options().dtype(acc_type));
    bnrelu_calc_invstd(invstd, *running_var, epsilon);
  }

  Tensor sum_dy, sum_dy_xmu, grad_weight, grad_bias;
  if (needs_reduction) {
    std::tie(sum_dy, sum_dy_xmu, grad_weight, grad_bias) =
        bnrelu_backward_reduce_cuda(
            grad_out, input, output, mean, invstd, *weight,
            grad_input_mask[0], grad_input_mask[1], grad_input_mask[2]);
  }

  Tensor grad_input;
  if (grad_input_mask[0]) {
    if (train) {
      // NOTE: sum_dy and sum_dy_xmy are defined, as train implies needs_reduction
      grad_input = bnrelu_elementwise_backward_train(
          grad_out, input, output, mean, invstd, *weight, sum_dy, sum_dy_xmu);
    } else {
      grad_input = bnrelu_elementwise_backward_eval(
          grad_out, input, output, invstd, *weight);
    }
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor> bnrelu_stats_cuda(const Tensor& self, double epsilon) {
  auto options = self.options().dtype(
      at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true));
  auto n_channels = self.size(1);
  auto save_mean = at::empty({n_channels}, options);
  auto save_invstd = at::empty({n_channels}, options);

  bool use_channels_last_kernel = bnrelu_use_channels_last_kernels(self);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  self.scalar_type(), "bnrelu_stats_cuda", [&] {
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (use_channels_last_kernel) {
        bnrelu_stats_channels_last_cuda_template<scalar_t, InvStd>(
            save_mean, save_invstd, self, epsilon);
      } else {
        bnrelu_stats_cuda_template<scalar_t, int32_t, InvStd>(
            save_mean, save_invstd, self, epsilon);
      }
    } else {
      bnrelu_stats_cuda_template<scalar_t, int64_t, InvStd>(
          save_mean, save_invstd, self, epsilon);
    }
  });
  return std::tuple<Tensor, Tensor>(save_mean, save_invstd);
}
#if 0
Tensor bnrelu_elemt_cuda(
    const Tensor& self, const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt, const Tensor& mean,
    const Tensor& invstd, double epsilon) {
  auto output = at::empty_like(self);
  // FIXME: Epsilon parameter isn't required, we don't take the reciprocal
  bnrelu_elementwise(output, self, weight_opt, bias_opt, mean, invstd);
  return output;
}

Tensor& bnrelu_elemt_cuda_out(const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt,
                                  const Tensor& mean, const Tensor& invstd, double epsilon, Tensor& output) {
  // FIXME: Epsilon parameter isn't required, we don't take the reciprocal
  bnrelu_elementwise(output, self, weight_opt, bias_opt, mean, invstd);
  return output;
}
#endif
// accepting input(self) here to determine template data types, since running_mean/running_var are optional
std::tuple<Tensor, Tensor> bnrelu_gather_stats_cuda(const Tensor& self, const Tensor& mean, const Tensor& invstd, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, double momentum, double epsilon, int64_t count) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  std::vector<int64_t> counts(mean.size(0), count);
  Tensor counts_ = at::from_blob((void*)counts.data(), {(int64_t)counts.size()}, self.options().dtype(at::kLong).device(at::kCPU));
  counts_ = counts_.to(self.device()).to(running_mean.defined() ? running_mean.dtype() : self.dtype());
  return bnrelu_gather_stats_with_counts_cuda(self, mean, invstd, running_mean, running_var, momentum, epsilon, counts_);
}


std::tuple<Tensor, Tensor> bnrelu_gather_stats_with_counts_cuda(
    const Tensor& self, const Tensor& mean, const Tensor& invstd, const c10::optional<Tensor>& running_mean_opt /* optional */, const c10::optional<Tensor>& running_var_opt /* optional */, double momentum, double epsilon, const Tensor& counts) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});


  auto scalar_type = running_mean.defined() ? running_mean.scalar_type() : self.scalar_type();
  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "bnrelu_update_stats_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      return bnrelu_gather_stats_cuda_template<scalar_t, accscalar_t, int32_t>(mean, invstd, running_mean, running_var, momentum, epsilon, counts);
    } else {
      return bnrelu_gather_stats_cuda_template<scalar_t, accscalar_t, int64_t>(mean, invstd, running_mean, running_var, momentum, epsilon, counts);
    }
  });
}

std::tuple<Tensor, Tensor, Tensor, Tensor> bnrelu_backward_reduce_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& output, const Tensor& mean, const Tensor& invstd, const c10::optional<Tensor>& weight_opt, bool input_g, bool weight_g, bool bias_g) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  if (at::cuda::detail::canUse32BitIndexMath(grad_output) &&
      bnrelu_use_channels_last_kernels(grad_output) &&
      bnrelu_use_channels_last_kernels(input) &&
      (!weight.defined() || weight.is_contiguous()) &&
      mean.is_contiguous() && invstd.is_contiguous()){
    return bnrelu_backward_reduce_cuda_channels_last_template(
        grad_output, input, output, mean, invstd, weight, input_g, weight_g, bias_g);
  }

  return AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, grad_output.scalar_type(), "bnrelu_backward_reduce", [&] {
    auto mean_st = mean.dtype();
    auto invstd_st = invstd.dtype();
    TORCH_CHECK(mean_st == invstd_st, "mean and invstd need to have the same data types");
    const bool mixed_type = is_mixed_type(input, weight);
    using accscalar_t = at::acc_type<scalar_t, true>;

    if (cuda::detail::canUse32BitIndexMath(grad_output)) {
      if (mixed_type) {
        return bnrelu_backward_reduce_cuda_template<scalar_t, accscalar_t, int32_t>(grad_output, input, output, mean, invstd, weight, input_g, weight_g, bias_g);
      } else {
        return bnrelu_backward_reduce_cuda_template<scalar_t, scalar_t, int32_t>(grad_output, input, output, mean, invstd, weight, input_g, weight_g, bias_g);
      }
    } else {
      if (mixed_type) {
        return bnrelu_backward_reduce_cuda_template<scalar_t, accscalar_t, int64_t>(grad_output, input, output, mean, invstd, weight, input_g, weight_g, bias_g);
      } else {
        return bnrelu_backward_reduce_cuda_template<scalar_t, scalar_t, int64_t>(grad_output, input, output, mean, invstd, weight, input_g, weight_g, bias_g);
      }
    }
  });
}

Tensor bnrelu_backward_elemt_cuda(const Tensor& self, const Tensor& input, const Tensor& output, const Tensor& mean, const Tensor& invstd, const c10::optional<Tensor>& weight_opt, const Tensor& sum_dy, const Tensor& sum_dy_xmu, const Tensor& count) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  if (at::cuda::detail::canUse32BitIndexMath(self) &&
      bnrelu_use_channels_last_kernels(self) &&
      bnrelu_use_channels_last_kernels(input))  {
    return bnrelu_backward_elemt_channels_last_cuda_template(self, input, output, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
  }

  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "bnrelu_backward_elemt", [&] {
    auto mean_st = mean.dtype();
    auto invstd_st = invstd.dtype();
    TORCH_CHECK(mean_st == invstd_st, "mean and invstd need to have the same data types");
    bool is_half_float = std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
    bool is_bfloat16_float = std::is_same<scalar_t, at::BFloat16>::value && mean_st == at::kFloat;
    using accscalar_t = at::acc_type<scalar_t, true>;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (is_half_float || is_bfloat16_float) {
        return bnrelu_backward_elemt_cuda_template<scalar_t, accscalar_t, int32_t>(self, input, output, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
      } else {
        return bnrelu_backward_elemt_cuda_template<scalar_t, scalar_t, int32_t>(self, input, output, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
      }
    } else {
      if (is_half_float || is_bfloat16_float) {
        return bnrelu_backward_elemt_cuda_template<scalar_t, accscalar_t, int64_t>(self, input, output, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
      } else {
        return bnrelu_backward_elemt_cuda_template<scalar_t, scalar_t, int64_t>(self, input, output, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
      }
    }
  });
}

std::tuple<Tensor, Tensor> bnrelu_update_stats_cuda(
    const Tensor& self, const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt, double momentum) {
  c10::MaybeOwned<Tensor> running_mean = at::borrow_from_optional_tensor(running_mean_opt);
  c10::MaybeOwned<Tensor> running_var = at::borrow_from_optional_tensor(running_var_opt);

  const int64_t n_input = self.size(1);
  auto options = self.options().dtype(
      at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true));
  auto save_mean = at::empty({n_input}, options);
  auto save_var = at::empty({n_input}, options);

  bnrelu_mean_var(self, save_mean, save_var);
  TORCH_CHECK(running_mean->defined() == running_var->defined());
  if (running_mean->defined()) {
    const int64_t N = self.numel() / save_mean.numel();
    bnrelu_update_stats(save_mean, save_var, *running_mean, *running_var, momentum, N);
  }
  return std::tuple<Tensor, Tensor>(save_mean, save_var);
}
//up code is native
//bnrelu check(miopen/native)
//forward
size_t get_env_miopen(const char* env_name, size_t def_value = 0) 
{
  if (auto* value = std::getenv(env_name)) {
    int env_value = c10::stoi(value);
    if (env_value != 1) return def_value;
    return env_value;
  }
  return def_value;
}
std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t> _bnrelu_impl_index(
    const Tensor& input, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, 
    const c10::optional<Tensor>& running_var_opt, bool training, double momentum, double epsilon, int fuse_act_mode) {

  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  auto num_features = input.sizes()[1]; 
  if (running_mean.defined()) {
    check_dims_match_num_input_features("running_mean", num_features, running_mean.numel());
  } else if (!training) {
    AT_ERROR("running_mean must be defined in evaluation mode");
  }
  if (running_var.defined()) {
    check_dims_match_num_input_features("running_var", num_features, running_var.numel());
  } else if (!training) {
    AT_ERROR("running_var must be defined in evaluation mode");
  }
  if (weight.defined()) {
    check_dims_match_num_input_features("weight", num_features, weight.numel());
  }
  if (bias.defined()) {
    check_dims_match_num_input_features("bias", num_features, bias.numel());
  }

  bool use_miopen = (input.is_cuda()
               && input.dim() <= 4 //ncdhw  miopen fuse is not used
               && input.scalar_type() != at::kDouble
               && input.scalar_type() != at::kBFloat16
               && (weight.scalar_type() != at::kHalf)
               && weight.defined() && bias.defined()
               && ((running_mean.defined() && running_var.defined())
                 || (!running_mean.defined() && !running_var.defined() && training))
               && detail::getCUDAHooks().compiledWithMIOpen()
               );
  uint32_t hw = 0;
  if (input.dim() == 4) {
    hw = input.size(2) * input.size(3);
  }
  if (get_env_miopen("HIP_NATIVE_FUSE") == 1) {
    goto use_native;
  }
  if (fuse_act_mode == fuse_act_silu)
    goto use_native;

  if ((use_miopen && hw <= 784) || (use_miopen && get_env_miopen("HIP_MIOPEN_FUSE")) == 1) {
    return std::tuple_cat(
               miopen_bnrelu_forward(
               input.contiguous(), weight.contiguous(), bias.contiguous(),
               running_mean.defined() ? running_mean.contiguous() : running_mean,
               running_var.defined() ? running_var.contiguous() : running_var,
               training, momentum, epsilon),
               std::make_tuple(2));
  }
use_native:
  return std::tuple_cat(
            bnrelu_cuda(input, weight, bias, running_mean, 
            running_var, training, momentum, epsilon, fuse_act_mode),
            std::make_tuple(0));
}

//bnrelu input
std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t> bnrelu_forward(
    const Tensor& input, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt,
    bool training, double momentum, double eps, int fuse_act_mode) {
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  return _bnrelu_impl_index(input, weight, bias, running_mean, running_var,
                                                training, momentum, eps, fuse_act_mode);
}
// backward
std::tuple<Tensor, Tensor, Tensor> bnrelu_backward(
    int64_t impl_index, const Tensor& grad_output, const Tensor& input, const Tensor& output_relu, const Tensor& output_relu_grad, 
    const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, 
    const c10::optional<Tensor>& running_var_opt, const c10::optional<Tensor>& save_mean_opt, const c10::optional<Tensor>& save_invstd_opt, 
    bool train, double epsilon, std::array<bool,3> output_mask, int fuse_act_mode) {
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  const Tensor& save_mean = c10::value_or_else(save_mean_opt, [] {return Tensor();});
  const Tensor& save_var_transform = c10::value_or_else(save_invstd_opt, [] {return Tensor();});
  
  if (input.numel() == 0) {
    std::vector<int64_t> dims(input.dim() - 1);
    dims[0] = 0;
    std::iota(dims.begin() + 1, dims.end(), 2);

    // don't return empty tensor because it will break gradient chain
    Tensor grad_input;
    Tensor grad_weight;
    Tensor grad_bias;
    if (output_mask[2]) {
      grad_bias = grad_output.sum(dims);
    }
    if (output_mask[1]) {
      grad_weight = (grad_output * input).sum(dims);
    }
    if (output_mask[0] && weight.defined()) {
      grad_input = grad_output * weight[0];
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
  }

  if (impl_index == 0 || (!train)) {
    return bnrelu_backward_cuda(grad_output, input, output_relu_grad, weight, running_mean, running_var, save_mean, save_var_transform, train, epsilon, output_mask, fuse_act_mode);
  } else if (impl_index == 2) {
    return miopen_bnrelu_backward(input, output_relu, grad_output, weight, bias, running_mean, running_var, save_mean, save_var_transform, epsilon);
  }
}

} } // namespace at::native
