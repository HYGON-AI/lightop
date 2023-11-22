#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cuda/Loops.cuh>
#ifdef TORCH_VERSION_OVER_110
    #include <ATen/native/cuda/Resize.h>
#else
    #include <ATen/native/cuda/Resize.cuh>
#endif
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/extension.h>
#include <fuse_bn_dropout.cuh>
namespace at { namespace native {

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

void check_dims_match_num_input_features(const char* arg_name, int64_t expected, int64_t actual){
    TORCH_CHECK(actual == expected,
             arg_name, " should contain ", expected, " elements not ", actual);
}

void bn_dropout_elementwise(
    const Tensor& out, const Tensor& out_dropout, const Tensor& self, const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt, const Tensor& mean_, const Tensor& invstd_, double p) {
  c10::MaybeOwned<Tensor> weight = at::borrow_from_optional_tensor(weight_opt);
  c10::MaybeOwned<Tensor> bias = at::borrow_from_optional_tensor(bias_opt);
  resize_output(out, self.sizes());
  resize_output(out_dropout, self.sizes());
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(),
                                  "bn_dropout_elementwise_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const bool mixed_type = is_mixed_type(self, *weight, *bias);
    if (mixed_type) {
      bn_dropout_elemt_cuda_template<scalar_t, accscalar_t, int32_t>(
          out, out_dropout, self, *weight, *bias, mean_, invstd_, p);
    } else {
      bn_dropout_elemt_cuda_template<scalar_t, scalar_t, int32_t>(
          out, out_dropout, self, *weight, *bias, mean_, invstd_, p);
    }
  });
  return;
 
}

void bn_dropout_mean_var(const Tensor& self, Tensor& save_mean, Tensor& save_var) {
  // NOTE: Epsilon is only used for InvStd, not Var. The value here is ignored.
  const double dummy_epsilon = 1e-5;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "bn_dropout_stats_cuda", [&] {
    bn_dropout_stats_cuda_template<scalar_t, int32_t, Var>(
        save_mean, save_var, self, dummy_epsilon);
  });
  return;
}

void bn_dropout_update_stats_and_invert(
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
                                  "bn_dropout_update_stats_cuda", [&] {
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

void bn_dropout_calc_invstd(const Tensor& out_invstd, const Tensor& running_var, double epsilon) {
  auto iter = TensorIteratorConfig()
      .add_output(out_invstd)
      .add_input(running_var)
      .check_all_same_dtype(false)
      .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, running_var.scalar_type(),
                                  "bn_dropout_invert_std_cuda", [&] {
    using acc_t = at::acc_type<scalar_t, true>;
    auto eps = static_cast<acc_t>(epsilon);
    gpu_kernel(iter, [eps] GPU_LAMBDA (scalar_t var) -> acc_t {
      return c10::cuda::compat::rsqrt(var + eps);
    });
  });
}
}

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> bn_dropout_cuda_out(const Tensor& self, const c10::optional<Tensor>& weight_opt, 
       const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, 
       bool train, double momentum, double epsilon, Tensor& output, Tensor& output_dropout, Tensor& save_mean, Tensor& save_invstd,
       double p) {
  const bool has_running_mean = (running_mean_opt.has_value() && running_mean_opt->defined());
  const bool has_running_var = (running_var_opt.has_value() && running_var_opt->defined());
  TORCH_CHECK(has_running_mean == has_running_var);
  p = 1 - p;
  if (train) {
    bn_dropout_mean_var(self, save_mean, save_invstd);
    if (has_running_mean) {
      const int64_t N = self.numel() / save_mean.numel();
      bn_dropout_update_stats_and_invert(
          save_mean, save_invstd, *running_mean_opt, *running_var_opt,
          momentum, epsilon, N);
    } else {
      bn_dropout_calc_invstd(save_invstd, save_invstd, epsilon);
    }
  } else {
    TORCH_CHECK(has_running_mean);
    at::native::resize_output(save_mean, running_mean_opt->sizes());
    save_mean.copy_(*running_mean_opt, /*non_blocking=*/true);
    bn_dropout_calc_invstd(save_invstd, running_var_opt.value(), epsilon);
  }

  bn_dropout_elementwise(output, output_dropout, self, weight_opt, bias_opt, save_mean, save_invstd, p);
  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>(output, output_dropout, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> bn_dropout_cuda(const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, 
      const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, bool train, double momentum, 
      double epsilon, double p) {
  auto output = at::empty_like(self);
  int64_t n_input = self.size(1);
  auto options = self.options().dtype(
      at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true));
  auto save_mean = at::empty({n_input}, options);
  auto save_invstd = at::empty({n_input}, options);
  Tensor output_dropout;
  output_dropout = at::empty_like(self, self.options().dtype(kBool));

  at::native::bn_dropout_cuda_out(
      self,
      weight_opt,
      bias_opt,
      running_mean_opt,
      running_var_opt,
      train,
      momentum,
      epsilon,
      output,
      output_dropout,
      save_mean,
      save_invstd,
      p);
  return std::make_tuple(output, output_dropout, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> bn_dropout_backward_cuda(const Tensor& grad_out, const Tensor& input, const Tensor& output_dropout, 
       const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, 
       const c10::optional<Tensor>& save_mean_opt, const c10::optional<Tensor>& save_invstd_opt, bool train, double epsilon, 
       std::array<bool,3> grad_input_mask, double p) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight = at::borrow_from_optional_tensor(weight_opt);
  c10::MaybeOwned<Tensor> save_mean = at::borrow_from_optional_tensor(save_mean_opt);
  c10::MaybeOwned<Tensor> save_invstd = at::borrow_from_optional_tensor(save_invstd_opt);
  c10::MaybeOwned<Tensor> running_mean = at::borrow_from_optional_tensor(running_mean_opt);
  c10::MaybeOwned<Tensor> running_var = at::borrow_from_optional_tensor(running_var_opt);
  const bool needs_reduction = train || grad_input_mask[1] || grad_input_mask[2];

  // Fused reducion & elementwise kernel
  // if (needs_reduction && grad_input_mask[0] &&
  //     // !bn_dropout_use_channels_last_kernels(input) &&
  //     cuda::detail::canUse32BitIndexMath(input) &&
  //     cuda::detail::canUse32BitIndexMath(grad_out)) {
    return AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                           "bn_dropout_backward_cuda", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      const bool mixed_type = is_mixed_type(input, *weight, *running_mean, *running_var);
      if (mixed_type) {
          return bn_dropout_backward_cuda_template<scalar_t, accscalar_t, int32_t>(
              grad_out, input, output_dropout, *weight, *running_mean, *running_var,
              *save_mean, *save_invstd, train, epsilon, grad_input_mask, p);
      } else {
          return bn_dropout_backward_cuda_template<scalar_t, scalar_t, int32_t>(
              grad_out, input, output_dropout, *weight, *running_mean, *running_var,
              *save_mean, *save_invstd, train, epsilon, grad_input_mask, p);
      }
    });
  // }
}

//forward
std::tuple<Tensor, Tensor, Tensor, Tensor> _bn_dropout_impl_index(
    const Tensor& input, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, 
    const c10::optional<Tensor>& running_var_opt, bool training, double momentum, double epsilon, double p) {

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

  return std::tuple_cat(
            bn_dropout_cuda(input, weight, bias, running_mean, 
            running_var, training, momentum, epsilon, p));
}

//bn_dropout input
std::tuple<Tensor, Tensor, Tensor, Tensor> bn_dropout_forward(
    const Tensor& input, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt,
    bool training, double momentum, double eps, double p) {
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  if (p == 1) {
    auto output = at::zeros_like(input);
    auto output_dropout = at::zeros_like(input, input.options().dtype(kBool));
    int64_t n_input = input.size(1);
    auto options = input.options().dtype(
                    at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true));
    auto save_mean = at::zeros({n_input}, options);
    auto save_invstd = at::zeros({n_input}, options);
    return std::make_tuple(output, output_dropout, save_mean, save_invstd);
  }
  return _bn_dropout_impl_index(input, weight, bias, running_mean, running_var, training, momentum, eps, p);
}
// backward
std::tuple<Tensor, Tensor, Tensor> bn_dropout_backward(const Tensor& grad_output, const Tensor& input, const Tensor& output_dropout,
    const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, 
    const c10::optional<Tensor>& running_var_opt, const c10::optional<Tensor>& save_mean_opt, const c10::optional<Tensor>& save_invstd_opt, 
    bool train, double epsilon, std::array<bool,3> output_mask, double p) {
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
  if (p == 1) {
    auto output = at::zeros_like(grad_output);
    int64_t n_input = grad_output.size(1);
    auto options = grad_output.options().dtype(
                      at::toAccumulateType(grad_output.scalar_type(), /*is_cuda=*/true));
    auto save_mean = at::zeros({n_input}, options);
    auto save_invstd = at::zeros({n_input}, options);
    return std::make_tuple(output, save_mean, save_invstd);
  }
  
  return bn_dropout_backward_cuda(grad_output, input, output_dropout, weight, running_mean, running_var, save_mean, save_var_transform, train, epsilon, output_mask, p);
}

} } // namespace at::native
