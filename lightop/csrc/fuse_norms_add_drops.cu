#include "fuse_norms_ln_rms_add_drops_device_kernel.cuh"

#include <string>

namespace at {
namespace native {

namespace {
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
}

namespace ln_rms_norms_add_drops {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// forward ////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename U=float, int VEC = 4>
__global__ void cuApply_LnRmsNorms_Forward(T *__restrict__ output_vals, U *__restrict__ mean, U *__restrict__ invvar,
                                           const T *__restrict__ input_vals,
                                           const int n1, const int n2, const U epsilon,
                                           const T *__restrict__ gamma, const T *__restrict__ beta, bool rms_only) {
    if (rms_only) {
        norms_device_kernel::cuApplyRmsNorm_<T, U, VEC>(output_vals, invvar, input_vals, n1, n2, epsilon, gamma);
    } else {
        norms_device_kernel::cuApplyLnNorm_<T, U, VEC>(output_vals, mean, invvar, input_vals, n1, n2, epsilon, gamma, beta);
    }
}

template<typename T, typename U=float, int VEC = 4>
__global__ void cuApply_AddDrops_Forward(T *__restrict__ output_vals, const T *__restrict__ input_vals,
                                         const int n1, const int n2,
                                         const T *__restrict__ residual, uint8_t *__restrict__ drop_mask,
                                         const U drop_keep_rate, const PhiloxCudaState philox_args) {
    if (drop_mask == nullptr) {
        add_drops_device_kernel::cuApplyAdd_<T, U, VEC>(output_vals, input_vals, residual, n1, n2);
    } else {
        add_drops_device_kernel::cuApplyAddDrops_<T, U, VEC>(output_vals, input_vals, residual, n1, n2,
                                                             drop_mask, drop_keep_rate, philox_args);
    }
}


template<typename T, typename U=float, int VEC = 4>
__global__ void cuApply_LnRmsNormsAddDrops_Forward(T *__restrict__ output_vals, U *__restrict__ mean, U *__restrict__ invvar,
                                                   const T *__restrict__ input_vals, T *__restrict__ temp_add_output,
                                                   const int n1, const int n2, const U epsilon,
                                                   const T *__restrict__ gamma, const T *__restrict__ beta, bool rms_only,
                                                   const T *__restrict__ residual, uint8_t *__restrict__ drop_mask,
                                                   const U drop_keep_rate, const PhiloxCudaState philox_args) {
    if (drop_mask == nullptr) {
        add_drops_device_kernel::cuApplyAdd_<T, U, VEC>(temp_add_output, input_vals, residual, n1, n2);
    } else {
        add_drops_device_kernel::cuApplyAddDrops_<T, U, VEC>(temp_add_output, input_vals, residual, n1, n2,
                                                             drop_mask, drop_keep_rate, philox_args);
    }

    if (rms_only) {
        norms_device_kernel::cuApplyRmsNorm_<T, U, VEC>(output_vals, invvar, temp_add_output, n1, n2, epsilon, gamma);
    } else {
        norms_device_kernel::cuApplyLnNorm_<T, U, VEC>(output_vals, mean, invvar, temp_add_output, n1, n2, epsilon, gamma, beta);
    }
}


template<typename T, typename U, int VEC = 4>
void HostApply_LnRmsNorm_AddDrop(T *output, U *mean, U *invvar, const T *input, T *temp_add_output,
                                 int n1, int n2, U epsilon,
                                 const T *gamma, const T *beta, bool rms_only,
                                 const T *residual, uint8_t *drop_mask, U drop_keep_rate, int runing_type_enum) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int warp_size = at::cuda::warp_size();
    const dim3 block_dims(warp_size, num_threads() / warp_size, 1);  // MI100 wavefront/warp = 64, num_threads()=256 on rocm
    const dim3 grid_dims(n1);
    // int VEC = 128 / sizeof(T);  // float -> VEC=4 ; half -> VEC=8

    // preparing add drop param
    int total_elements = n1 * n2;
    PhiloxCudaState philox_args;
    if (drop_mask != nullptr) {
        auto gen = get_generator_or_default<CUDAGeneratorImpl>(Generator(), cuda::detail::getDefaultCUDAGenerator());
        // counter in the random state
        int64_t counter_offset = ((total_elements - 1) / (total_elements * VEC) + 1) * VEC;
        {
            // See Note [Acquire lock when using random generators]
            std::lock_guard <std::mutex> lock(gen->mutex_);
            philox_args = gen->philox_cuda_state(counter_offset);
        }
    }

    // preparing norm param
    int nshared = 0;
    if (runing_type_enum <= 1) {   // use norm
        if (!rms_only) {
            nshared = block_dims.y * sizeof(U) + (block_dims.y / 2) * sizeof(U);
        } else {
            nshared = (block_dims.y / 2) * sizeof(U);
        }
    }
    if (runing_type_enum == 1) {
        cuApply_LnRmsNorms_Forward<T, U, VEC><<<grid_dims, block_dims, nshared, stream>>>(output, mean, invvar, input, n1, n2, epsilon, gamma, beta, rms_only);
    } else if (runing_type_enum == 2) {
        cuApply_AddDrops_Forward < T, U, VEC ><<<grid_dims, block_dims, nshared, stream>>>(output, input, n1, n2, residual, drop_mask, drop_keep_rate, philox_args);
    } else {
        cuApply_LnRmsNormsAddDrops_Forward < T, U, VEC ><<<grid_dims, block_dims, nshared, stream>>>(
            output, mean, invvar, input, temp_add_output, n1, n2, epsilon,
                gamma, beta, rms_only, residual, drop_mask, drop_keep_rate, philox_args);
    }
}

void cuda_ln_rms_norm_add_drops(int norm_type_enum, int bias_type_enum, int drop_type_enum,
                                at::Tensor *output, at::Tensor *mean, at::Tensor *invvar,
                                const at::Tensor *input, at::Tensor *temp_add_output,
                                int n1, int n2, at::Tensor *gamma, at::Tensor *beta, double epsilon, bool normalized_affine,
                                at::Tensor *residual, at::Tensor *drop_mask, double drop_prob, bool is_training, int runing_type_enum) {
    bool using_norm = norm_type_enum > 0;
    bool rms_only = norm_type_enum == 2;
    bool using_bias = bias_type_enum > 0;
    bool using_drop = drop_type_enum > 0 and is_training;
    bool using_bias_drop = using_bias || using_drop;

    using namespace at;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input->scalar_type(),
        "ln_rms_norm_add_drops_forward", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          HostApply_LnRmsNorm_AddDrop<scalar_t, accscalar_t, 4>(
              output->data_ptr<scalar_t>(),
              using_norm ? (rms_only ? nullptr : mean->data_ptr<accscalar_t>()) : nullptr,
              using_norm ? (invvar->data_ptr<accscalar_t>()) : nullptr,
              input->data_ptr<scalar_t>(),
              (runing_type_enum == 0) ? (temp_add_output->data_ptr<scalar_t>()) : nullptr,
              n1, n2, static_cast<accscalar_t>(epsilon),
              using_norm ? (normalized_affine ? gamma->data_ptr<scalar_t>() : nullptr) : nullptr,
              using_norm ? (rms_only ? nullptr : (normalized_affine ? beta->data_ptr<scalar_t>() : nullptr)) : nullptr,
              rms_only,
              using_bias ? (residual->data_ptr<scalar_t>()) : nullptr,
              using_drop ? (drop_mask->data_ptr<uint8_t>()) : nullptr,
              static_cast<accscalar_t>(1.f - drop_prob),
              runing_type_enum);
        }
    );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// backward ////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename U, int VEC = 4>
__global__ void cuApply_LnRmsNorms_Backward( /** only running norm */
    const T *__restrict__ dout,
    const T *__restrict__ input,
    const int n1, const int n2,
    const U *__restrict__ mean, const U *__restrict__ invvar, U epsilon,
    U *part_grad_gamma, U *part_grad_beta,
    T *grad_gamma, T *grad_beta,
    const T *gamma, bool rms_only, T *grad_input) {

    norms_device_kernel::cuApplyComputePartGradGammaBeta_<T, U>(dout, input, n1, n2, mean, invvar, epsilon, part_grad_gamma, part_grad_beta, rms_only);
    norms_device_kernel::cuApplyComputeGradGammaBeta_<T, U>(part_grad_gamma, part_grad_beta, n1, n2, grad_gamma, grad_beta, rms_only);
    norms_device_kernel::cuApplyComputeGradInput_<T, U, VEC>(grad_input, dout, input, n1, n2, mean, invvar, epsilon, gamma, rms_only);
}


template<typename T, typename U, int VEC = 4>
__global__ void cuApply_AddDrops_Backward( /** only running add_drop */
    const T *__restrict__ dout,
    const int n1, const int n2,
    T *grad_input, T *grad_residual,
    const uint8_t *drop_mask, U pinv) {
    if (drop_mask == nullptr) {
        add_drops_device_kernel::cuApplyAddGradient_<T, U, VEC>(grad_input, grad_residual, dout, n1, n2);
    } else {
        add_drops_device_kernel::cuApplyAddDropsGradient_<T, U, VEC>(grad_input, grad_residual, dout, drop_mask, n1, n2, pinv);
    }
}

template<typename T, typename U, int VEC = 4>
__global__ void cuApply_LnRmsAddDrops_Backward( /** running add_drop and norm */
    const T *__restrict__ dout,
    const T *__restrict__ input,
    const int n1, const int n2,
    const U *__restrict__ mean, const U *__restrict__ invvar,
    U epsilon,
    U *part_grad_gamma, U *part_grad_beta,
    T *grad_gamma, T *grad_beta,
    const T *gamma,
    bool rms_only,
    T *grad_input, T *grad_residual, T *temp_grad,
    const uint8_t *drop_mask, U pinv) {

    norms_device_kernel::cuApplyComputePartGradGammaBeta_<T, U>(dout, input, n1, n2, mean, invvar, epsilon, part_grad_gamma, part_grad_beta, rms_only);
    norms_device_kernel::cuApplyComputeGradGammaBeta_<T, U>(part_grad_gamma, part_grad_beta, n1, n2, grad_gamma, grad_beta, rms_only);
    norms_device_kernel::cuApplyComputeGradInput_<T, U, VEC>(temp_grad, dout, input, n1, n2, mean, invvar, epsilon, gamma, rms_only);
    // add_drop backward
    if (drop_mask == nullptr) {
        add_drops_device_kernel::cuApplyAddGradient_<T, U, VEC>(grad_input, grad_residual, temp_grad, n1, n2);
    } else {
        add_drops_device_kernel::cuApplyAddDropsGradient_<T, U, VEC>(grad_input, grad_residual, temp_grad, drop_mask, n1, n2, pinv);
    }
}

template<typename T, typename U=float, int VEC = 4>
void HostApply_LnRmsNormAddDrop_Gradient(T *grad_input, T *grad_gamma, T *grad_beta,
                                         const T *dout, const U *mean, const U *invvar,
                                         const at::Tensor *input, T *temp_grad,
                                         int n1, int n2, const T *gamma, const T *beta, U epsilon,
                                         bool rms_only, bool normalized_affine,
                                         T *grad_residual, const uint8_t *drop_mask,
                                         U drop_pinv, int runing_type_enum) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int warp_size = at::cuda::warp_size();

    at::Tensor part_grad_gamma, part_grad_beta;
    if (invvar != nullptr) {
        const auto part_grad_dtype = (input->scalar_type() == at::ScalarType::Half || input->scalar_type() == at::ScalarType::BFloat16) ?
                                     at::ScalarType::Float : input->scalar_type();
        if (normalized_affine) {
            part_grad_gamma = at::empty({warp_size, n2}, input->options().dtype(part_grad_dtype).requires_grad(false));
            if (!rms_only) {
                part_grad_beta = at::empty_like(part_grad_gamma);
            }
        }
    }

    const dim3 block_dims(warp_size, num_threads() / warp_size, 1);  // MI100 wavefront/warp = 64
    const dim3 grid_dims(n1);

    // compute grad_gamma(j) and grad_beta(j) and compute grad_input
    int nshared = 0;
    if (invvar != nullptr) {
        const int nshared_a = 2 * sizeof(U) * block_dims.y * block_dims.y * (block_dims.x + 1);
        const int nshared_b = block_dims.x * block_dims.y * sizeof(U);
        nshared = std::max(nshared_a, nshared_b);
    }

    if (runing_type_enum == 1) {
        cuApply_LnRmsNorms_Backward < T, U, VEC ><<<grid_dims, block_dims, nshared, stream>>>(
            dout, input->data_ptr<T>(), n1, n2, mean, invvar, epsilon,
                (invvar != nullptr) ? (normalized_affine ? part_grad_gamma.data_ptr<U>() : nullptr) : nullptr,
                (invvar != nullptr) ? (normalized_affine ? (rms_only ? nullptr : part_grad_beta.data_ptr<U>()) : nullptr) : nullptr,
                grad_gamma, grad_beta, gamma, rms_only, grad_input);
    } else if (runing_type_enum == 2) {
        cuApply_AddDrops_Backward < T, U, VEC ><<<grid_dims, block_dims, nshared, stream>>>(
            dout, n1, n2, grad_input, grad_residual, drop_mask, drop_pinv);
    } else {
        cuApply_LnRmsAddDrops_Backward < T, U, VEC ><<<grid_dims, block_dims, nshared, stream>>>(
            dout, input->data_ptr<T>(), n1, n2, mean, invvar, epsilon,
                (invvar != nullptr) ? (normalized_affine ? part_grad_gamma.data_ptr<U>() : nullptr) : nullptr,
                (invvar != nullptr) ? (normalized_affine ? (rms_only ? nullptr : part_grad_beta.data_ptr<U>()) : nullptr) : nullptr,
                grad_gamma, grad_beta, gamma, rms_only, grad_input, grad_residual, temp_grad, drop_mask, drop_pinv);
    }
}

void cuda_ln_rms_norm_add_drops_gradient(int norm_type_enum, int bias_type_enum, int drop_type_enum,
                                         at::Tensor *dout, at::Tensor *mean, at::Tensor *invvar, at::Tensor *input, at::Tensor *temp_grad,
                                         int n1, int n2, at::Tensor *gamma, at::Tensor *beta, double epsilon,
                                         at::Tensor *grad_input, at::Tensor *grad_gamma, at::Tensor *grad_beta, bool normalized_affine,
                                         at::Tensor *grad_residual, at::Tensor *drop_mask, double drop_prob, bool is_training, int runing_type_enum) {
    bool using_norm = norm_type_enum > 0;
    bool rms_only = norm_type_enum == 2;
    bool using_bias = bias_type_enum > 0;
    bool using_drop = drop_type_enum > 0 and is_training;
    bool using_bias_drop = using_bias || using_drop;

    using namespace at;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input->scalar_type(),
        "ln_rms_norm_add_drops_gradient_backward", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          HostApply_LnRmsNormAddDrop_Gradient<scalar_t, accscalar_t, 4>(
              grad_input->data_ptr<scalar_t>(),
              using_norm ? (normalized_affine ? grad_gamma->data_ptr<scalar_t>() : nullptr) : nullptr,
              using_norm ? (rms_only ? nullptr : (normalized_affine ? grad_beta->data_ptr<scalar_t>() : nullptr)) : nullptr,
              dout->data_ptr<scalar_t>(),
              using_norm ? (rms_only ? nullptr : mean->data_ptr<accscalar_t>()) : nullptr,
              using_norm ? (invvar->data_ptr<accscalar_t>()) : nullptr,
              input,
              (runing_type_enum == 0) ? (temp_grad->data_ptr<scalar_t>()) : nullptr,
              n1, n2,
              normalized_affine ? gamma->data_ptr<scalar_t>() : nullptr,
              using_norm ? (rms_only ? nullptr : (normalized_affine ? beta->data_ptr<scalar_t>() : nullptr)) : nullptr,
              static_cast<accscalar_t>(epsilon),
              rms_only,
              normalized_affine,
              using_bias ? grad_residual->data_ptr<scalar_t>() : nullptr,
              using_drop ? drop_mask->data_ptr<uint8_t>() : nullptr,
              static_cast<accscalar_t>(1 / (1 - drop_prob)),
              runing_type_enum);
        }
    );
}


} // ln_rms_norms_add_drops

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 调用接口
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector <at::Tensor> lnRmsNorms_add_drops_forward(
    int norm_type_enum, int bias_type_enum, int drop_type_enum,
    at::Tensor input, int normalized_shape,
    at::Tensor weight, at::Tensor bias, double epsilon, bool normalized_affine,
    at::Tensor residual, double drop_prob, bool is_training) {
    /** compute Norms( drop(x) + bias )
     * norm_type_enum: norm type  none/ln/rms
     * bias_type_enum: none/bias
     * drop_type_enum: none/dropout/droppath
     * input: input tensor, x
     * normalized_shape: normalized shape for norm
     * weight: norm param
     * bias: norm param
     * epsilon: norm param
     * normalized_affine: norm has weight or bias
     * residual: bias tensor
     * drop_prob: drop rate of drop
     * is_training: drop running only training
     * */
    bool using_norm = norm_type_enum > 0;
    bool using_bias = bias_type_enum > 0;
    bool using_drop = drop_type_enum > 0 and is_training;
    bool using_bias_drop = using_bias || using_drop;

    int runing_type_enum;
    if (!using_norm && using_bias_drop) {
        runing_type_enum = 2;
    } else if (using_norm && !using_bias_drop) {
        runing_type_enum = 1;
    } else if (using_norm && using_bias_drop) {
        runing_type_enum = 0;
    } else {
        std::stringstream ss;
        ss << "using norm or using add drop\n";
        throw std::runtime_error(ss.str());
    }
    //////////////////////////////////////////// prepare norms ////////////////////////////////////////////
    // Compared to contiguous, it saves a reference count increment/decrement if *this is already contiguous,
    // at the cost in all cases of an extra pointer of stack usage, an extra branch to access, and an extra branch at destruction time.
    auto X_in = input.expect_contiguous();
    auto dst_options = X_in->options().requires_grad(false);
    at::Tensor output = torch::empty(X_in->sizes(), dst_options);

    at::Tensor invvar, mean, temp_add_output;

    int n1, n2;
    compute_n1_n2(*X_in, normalized_shape, n1, n2);

    if (using_norm) { // mean and sigma2 for norm
        invvar = at::empty({n1}, dst_options);
        if (norm_type_enum == 1) {
            mean = at::empty({n1}, dst_options);
        }
    }

    if (runing_type_enum == 0) { // temp output of add drops, used by norm
        temp_add_output = at::native::empty_like(*X_in,
                                                 c10::nullopt /* dtype */,
                                                 c10::nullopt /* layout */,
                                                 c10::nullopt /* device */,
                                                 c10::nullopt /* pin_memory */,
                                                 LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }

    //////////////////////////////////////////// prepare add drops ////////////////////////////////////////////
    if (using_bias) {
        CHECK_INPUT(residual);
        TORCH_CHECK(X_in->sizes() == residual.sizes(), "input residual must has same shape");
    }

    auto mask_options = X_in->options().requires_grad(false).dtype(torch::kUInt8);
    torch::Tensor drop_mask;
    if (using_drop) {
        TORCH_CHECK(drop_prob >= 0 && drop_prob <= 1, "drop prob is during 0 and 1");
        drop_mask = torch::empty_like(input, mask_options);
    } else {
        drop_mask = torch::empty({1}, mask_options);
    }

    //////////////////////////////////////////// prepare add drops ////////////////////////////////////////////
    ln_rms_norms_add_drops::cuda_ln_rms_norm_add_drops(
        norm_type_enum, bias_type_enum, drop_type_enum,
        &output, &mean, &invvar, &input, &temp_add_output,
        n1, n2, &weight, &bias, epsilon, normalized_affine,
        &residual, &drop_mask, drop_prob, is_training, runing_type_enum);

    // if (!drop_mask.numel()) { ("drop_mask is nullptr\n"); } else { ("drop_mask is not nullptr\n"); }
    if (runing_type_enum == 0) {
        return {output, mean, invvar, temp_add_output, drop_mask};
    } else {
        return {output, mean, invvar, input, drop_mask};
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector <at::Tensor> lnRmsNorms_add_drops_backward(
    int norm_type_enum, int bias_type_enum, int drop_type_enum,
    at::Tensor dout, at::Tensor mean, at::Tensor invvar, at::Tensor input, int normalized_shape,
    at::Tensor weight, at::Tensor bias, double epsilon, bool normalized_affine,
    at::Tensor drop_mask, double drop_prob, bool is_training) {

    bool using_norm = norm_type_enum > 0;
    bool using_bias = bias_type_enum > 0;
//    bool using_drop = (drop_type_enum > 0) && is_training && (drop_prob > 0);    // !!! use this after drop is correct
    bool using_drop = (drop_type_enum > 0) && is_training;
    bool using_bias_drop = using_bias || using_drop;
    int runing_type_enum;
    if (!using_norm && using_bias_drop) {
        runing_type_enum = 2;
    } else if (using_norm && !using_bias_drop) {
        runing_type_enum = 1;
    } else if (using_norm && using_bias_drop) {
        runing_type_enum = 0;
    } else {
        std::stringstream ss;
        ss << "using norm or using add drop\n";
        throw std::runtime_error(ss.str());
    }

    auto X_dout = dout.expect_contiguous();
    // check tensors
    CHECK_INPUT(dout);
    if (using_norm) {
        CHECK_INPUT(invvar);
    }
    if (using_drop) { CHECK_INPUT(drop_mask); }

    // prapare tensors
    at::Tensor grad_input = at::native::empty_like(*X_dout, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    at::Tensor grad_gamma, grad_beta, temp_grad;
    if (using_norm) {
        grad_gamma = at::empty_like(weight);
        grad_beta = at::empty_like(bias);
    }
    if (runing_type_enum == 0) {
        temp_grad = at::native::empty_like(*X_dout, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    at::Tensor grad_residual;
    if (using_bias) { grad_residual = at::empty_like(grad_input); }

    int n1, n2;
    compute_n1_n2(*X_dout, normalized_shape, n1, n2);

    ln_rms_norms_add_drops::cuda_ln_rms_norm_add_drops_gradient(
        norm_type_enum, bias_type_enum, drop_type_enum,
        &dout, &mean, &invvar, &input, &temp_grad,
        n1, n2, &weight, &bias, epsilon,
        &grad_input, &grad_gamma, &grad_beta, normalized_affine,
        &grad_residual, &drop_mask, drop_prob, is_training, runing_type_enum);

    return {grad_input, grad_gamma, grad_beta, grad_residual};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FusedLnRmsNormAddDropFunction : public torch::autograd::Function<FusedLnRmsNormAddDropFunction> {
public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext *ctx,
      const torch::autograd::Variable &x,
      const torch::autograd::Variable &weight,
      const torch::autograd::Variable &bias,
      int normalized_shape,
      double normalized_eps,
      bool normalized_affine,
      const torch::autograd::Variable &residual,
      double drop_prob,
      int norm_type_enum, int bias_type_enum, int drop_type_enum,
      bool training) {

      ctx->saved_data["norm_type_enum"] = norm_type_enum;
      ctx->saved_data["bias_type_enum"] = bias_type_enum;
      ctx->saved_data["drop_type_enum"] = drop_type_enum;
      ctx->saved_data["normalized_shape"] = normalized_shape;
      ctx->saved_data["normalized_eps"] = normalized_eps;
      ctx->saved_data["normalized_affine"] = normalized_affine;
      ctx->saved_data["drop_prob"] = drop_prob;
      ctx->saved_data["training"] = training;

      // ret: output, mean, invvar, temp_add_output, drop_mask
      // ctx.save_for_backward(weight, bias, mean, invvar, temp_add_output, drop_mask)
      auto ret = lnRmsNorms_add_drops_forward(norm_type_enum, bias_type_enum, drop_type_enum,
                                              x, normalized_shape, weight, bias, normalized_eps, normalized_affine,
                                              residual, drop_prob, training);
      ctx->save_for_backward({weight, bias, ret[1], ret[2], ret[3], ret[4]});

      return {ret[0], ret[4]};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      const torch::autograd::variable_list &gradOutput) {
      auto saved = ctx->get_saved_variables();

      int norm_type_enum = ctx->saved_data["norm_type_enum"].toInt();
      int bias_type_enum = ctx->saved_data["bias_type_enum"].toInt();
      int drop_type_enum = ctx->saved_data["drop_type_enum"].toInt();
      int normalized_shape = ctx->saved_data["normalized_shape"].toInt();
      double normalized_eps = ctx->saved_data["normalized_eps"].toDouble();
      bool normalized_affine = ctx->saved_data["normalized_affine"].toBool();
      double drop_prob = ctx->saved_data["drop_prob"].toDouble();
      bool training = ctx->saved_data["training"].toBool();

      // weight, bias, mean, invvar, temp_add_output, drop_mask = ctx.saved_tensors
      // ret: grad_input, grad_weight, grad_bias, grad_residual
      // ctx->save_for_backward({weight, bias, ret[1], ret[2], ret[3], ret[4]});
      auto ret = lnRmsNorms_add_drops_backward(norm_type_enum, bias_type_enum, drop_type_enum,
                                               gradOutput[0], saved[2], saved[3], saved[4], normalized_shape,
                                               saved[0], saved[1], normalized_eps, normalized_affine, saved[5], drop_prob, training);
      return {ret[0], ret[1], ret[2],
              torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable(),
              ret[3],
              torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable()};
  }
};

std::tuple <Tensor, Tensor> fuse_lnrmsnorms_add_drop_forward_autograd(const Tensor &x,
                                                                      const Tensor &weight, const Tensor &bias,
                                                                      int normalized_shape, double normalized_eps, bool normalized_affine,
                                                                      const Tensor &residual,
                                                                      double drop_prob,
                                                                      int norm_type_enum, int bias_type_enum, int drop_type_enum,
                                                                      bool training) {
    auto result = FusedLnRmsNormAddDropFunction::apply(x, weight, bias, normalized_shape, normalized_eps, normalized_affine,
                                                       residual, drop_prob, norm_type_enum, bias_type_enum, drop_type_enum, training);
    return std::tuple<Tensor, Tensor>(result[0], result[1]);
}

} // native
} // at
