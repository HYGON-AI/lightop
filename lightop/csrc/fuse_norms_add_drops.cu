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

constexpr int block_size_VEC8 = 512;
constexpr int FUSED_LNRMSNORMS_BLOCK_SIZE = 256;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// forward ////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename U, int VEC = 4>
__global__ void cuApply_LnRmsNorms_Forward(T *__restrict__ output_vals, U *__restrict__ mean, U *__restrict__ rstd,
                                           const T *__restrict__ input_vals,
                                           const int n1, const int n2, const U epsilon,
                                           const T *__restrict__ gamma, const T *__restrict__ beta, bool rms_only) {
    if (!rms_only) {
        norms_device_kernel::cuApplyLnNorm_<T, U, VEC>(output_vals, mean, rstd, input_vals, n1, n2, epsilon, gamma, beta);
    } else {
        norms_device_kernel::cuApplyRmsNorm_<T, U, VEC>(output_vals, rstd, input_vals, n1, n2, epsilon, gamma);
    }
}

template<typename T, typename U, int VEC = 4>
__global__ void cuApply_AddDrops_Forward(T *__restrict__ output_vals, const T *__restrict__ input_vals,
                                         const int n1, const int n2,
                                         const T *__restrict__ residual, uint8_t *__restrict__ drop_mask,
                                         const U drop_keep_rate, const PhiloxCudaState philox_args) {
    // already assert drop_mask!=nullptr or residual!=nullptr
    if (drop_mask == nullptr) {
        add_drops_device_kernel::cuApplyAdd_<T, U, VEC>(output_vals, input_vals, residual, n1, n2);
    } else {
        add_drops_device_kernel::cuApplyAddDrops_<T, U, VEC>(output_vals, input_vals, residual, n1, n2,
                                                             drop_mask, drop_keep_rate, philox_args);
    }
}

template<typename T, typename U, int VEC = 4>
__global__ void cuApply_LnRmsNormsAddDrops_Forward(T *__restrict__ output_vals, U *__restrict__ mean, U *__restrict__ rstd,
                                                   const T *__restrict__ input_vals, T *__restrict__ temp_add_output,
                                                   const int n1, const int n2, const U epsilon,
                                                   const T *__restrict__ gamma, const T *__restrict__ beta, bool rms_only,
                                                   const T *__restrict__ residual, uint8_t *__restrict__ drop_mask,
                                                   const U drop_keep_rate, const PhiloxCudaState philox_args) {
    // already assert drop_mask!=nullptr or residual!=nullptr
    if (drop_mask == nullptr) {
        add_drops_device_kernel::cuApplyAdd_<T, U, VEC>(temp_add_output, input_vals, residual, n1, n2);
    } else {
        add_drops_device_kernel::cuApplyAddDrops_<T, U, VEC>(temp_add_output, input_vals, residual, n1, n2,
                                                             drop_mask, drop_keep_rate, philox_args);
    }

    if (!rms_only) {
        norms_device_kernel::cuApplyLnNorm_<T, U, VEC>(output_vals, mean, rstd, temp_add_output, n1, n2, epsilon, gamma, beta);
    } else {
        norms_device_kernel::cuApplyRmsNorm_<T, U, VEC>(output_vals, rstd, temp_add_output, n1, n2, epsilon, gamma);
    }
}

template<typename T, typename U, int VEC = 4>
void HostApply_LnRmsNorm_AddDrop(T *output, U *mean, U *rstd, const T *X, T *temp_add_output,
                                 int n1, int n2, U epsilon,
                                 const T *gamma, const T *beta, bool rms_only,
                                 const T *residual, uint8_t *drop_mask, U drop_keep_rate, int runing_type_enum) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int warp_size = at::cuda::warp_size();
    const dim3 block_dims(warp_size, FUSED_LNRMSNORMS_BLOCK_SIZE / warp_size, 1);  // MI100 wavefront/warp = 64, FUSED_LNRMSNORMS_BLOCK_SIZE=256 on rocm
    const dim3 grid_dims(n1);
    // int VEC = 128 / sizeof(T);

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
    if (rstd != nullptr) {   // use norm
        if (!rms_only) {
            nshared = block_dims.y * sizeof(U) + (block_dims.y / 2) * sizeof(U);
        } else {
            nshared = (block_dims.y / 2) * sizeof(U);
        }
    }
    if (runing_type_enum == 1) {
        cuApply_LnRmsNorms_Forward<T, U, VEC><<<grid_dims, block_dims, nshared, stream>>>(output, mean, rstd, X, n1, n2, epsilon, gamma, beta, rms_only);
    } else if (runing_type_enum == 2) {
        cuApply_AddDrops_Forward < T, U, VEC ><<<grid_dims, block_dims, nshared, stream>>>(output, X, n1, n2, residual, drop_mask, drop_keep_rate, philox_args);
    } else {
        cuApply_LnRmsNormsAddDrops_Forward < T, U, VEC ><<<grid_dims, block_dims, nshared, stream>>>(
            output, mean, rstd, X, temp_add_output, n1, n2, epsilon,
                gamma, beta, rms_only, residual, drop_mask, drop_keep_rate, philox_args);
    }
}

void cuda_ln_rms_norm_add_drops(int norm_type_enum, bool use_residual, bool use_dropout,
                                at::Tensor *output, at::Tensor *mean, at::Tensor *rstd,
                                const c10::MaybeOwned <Tensor> X, at::Tensor *temp_add_output,
                                int n1, int n2,
                                const at::Tensor *gamma, const at::Tensor *beta,
                                double epsilon, bool normalized_affine,
                                const at::Tensor *residual, at::Tensor *drop_mask,
                                double drop_prob, bool is_training, int runing_type_enum) {
    bool use_norm = norm_type_enum > 0;
    bool rms_only = norm_type_enum == 2;
    bool use_drop = use_dropout && is_training && (drop_prob > 0);
    bool use_residual_drop = use_residual || use_drop;

    using namespace at;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        X->scalar_type(),
        "ln_rms_norm_add_drops_forward", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>; // half,float; half,half

          HostApply_LnRmsNorm_AddDrop<scalar_t, accscalar_t, 4>(
              output->data_ptr<scalar_t>(),
              use_norm ? (rms_only ? nullptr : mean->data_ptr<accscalar_t>()) : nullptr,
              use_norm ? (rstd->data_ptr<accscalar_t>()) : nullptr,
              X->data_ptr<scalar_t>(),
              (runing_type_enum == 0) ? (temp_add_output->data_ptr<scalar_t>()) : nullptr,
              n1, n2, static_cast<accscalar_t>(epsilon),
              use_norm ? (normalized_affine ? gamma->data_ptr<scalar_t>() : nullptr) : nullptr,
              use_norm ? (rms_only ? nullptr : (normalized_affine ? beta->data_ptr<scalar_t>() : nullptr)) : nullptr,
              rms_only,
              use_residual ? (residual->data_ptr<scalar_t>()) : nullptr,
              use_drop ? (drop_mask->data_ptr<uint8_t>()) : nullptr,
              static_cast<accscalar_t>(1.f - drop_prob),
              runing_type_enum);
        }
    );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// backward ////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename U, int VEC = 4>
__global__ void cuApply_LnRmsNorms_gradInput_Backward( /** only running norm */
    const T *__restrict__ dY, const T *__restrict__ X,
    const U *__restrict__ mean, const U *__restrict__ rstd,
    const T *__restrict__ gamma,
    T *grad_input,
    const int n1, const int n2,
    bool rms_only) {

    extern __shared__ char s_data1[];
    U *buf = reinterpret_cast<U *>(&s_data1);

    if (!rms_only) {
        norms_device_kernel::cuApply_Ln_GradInput_<T, U, VEC>(dY, X, mean, rstd, gamma, grad_input, n1, n2, buf);
    } else {
        norms_device_kernel::cuApply_Rms_GradInput_<T, U, VEC>(dY, X, rstd, gamma, grad_input, n1, n2, buf);
    }
}

template<typename T, typename U, int VEC = 4>
__global__ void cuApply_AddDrops_Backward( /** only running add_drop */
    const T *__restrict__ dY,
    const int n1, const int n2,
    T *grad_input, T *grad_residual,
    const uint8_t *drop_mask, U pinv) {
    // already assert drop_mask!=nullptr, and residual checkout
    add_drops_device_kernel::cuApplyAddDropsGradient_<T, U, VEC>(grad_input, grad_residual, dY, drop_mask, n1, n2, pinv);
}

template<typename T, typename U, int VEC = 4>
__global__ void cuApply_LnRmsNormsAddDrops_gradInput_Backward( /** running add_drop and norm */
    const T *__restrict__ dY, const T *__restrict__ X,
    const U *__restrict__ mean, const U *__restrict__ rstd,
    const T *__restrict__ gamma,
    const int n1, const int n2, bool rms_only,
    T *grad_input, T *grad_residual, T *temp_grad,
    const uint8_t *drop_mask, U pinv) {

    alignas(VEC) extern __shared__ char s_data1[];
    U *buf = reinterpret_cast<U *>(&s_data1);

    if (!rms_only) {
        norms_device_kernel::cuApply_Ln_GradInput_<T, U, VEC>(dY, X, mean, rstd, gamma, temp_grad, n1, n2, buf);
    } else {
        norms_device_kernel::cuApply_Rms_GradInput_<T, U, VEC>(dY, X, rstd, gamma, temp_grad, n1, n2, buf);
    }

    // already assert drop_mask!=nullptr, and residual checkout
    add_drops_device_kernel::cuApplyAddDropsGradient_<T, U, VEC>(grad_input, grad_residual, temp_grad, drop_mask, n1, n2, pinv);
}

template<typename T, typename U, int VEC = 4>
void HostApply_LnRmsNorm_GammaBeta_Gradient(T *grad_gamma, T *grad_beta,
                                            const T *dY_data, const U *mean, const U *rstd, const T *X_data,
                                            int n1, int n2, bool rms_only) {
    // compute gamma beta for norm affine
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (n1 < block_size_VEC8) {
        // For small batch size, do colwise reduce directly.
        const dim3 block_dims(FUSED_LNRMSNORMS_BLOCK_SIZE);
        const dim3 grid_dims((n2 + FUSED_LNRMSNORMS_BLOCK_SIZE - 1) / FUSED_LNRMSNORMS_BLOCK_SIZE);
        if (!rms_only) {
            norms_device_kernel::cuApply_Ln_GammaBetaBackward_Simple<T, U> <<<grid_dims, block_dims, 0, stream>>>(
                n1, n2, dY_data, X_data, mean, rstd, grad_gamma, grad_beta);
        } else {
            norms_device_kernel::cuApply_Rms_GammaBackward_Simple<T, U> <<<grid_dims, block_dims, 0, stream>>>(
                n1, n2, dY_data, X_data, rstd, grad_gamma);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        const int thread_x = 16;
        dim3 threads{thread_x, 512 / thread_x};
        int blocks = (n2 + threads.x - 1) / threads.x;
        int nshared = 2 * sizeof(U) * threads.x * threads.y;
        if (!rms_only) {
            norms_device_kernel::cuApply_Ln_GammaBetaBackward_Vec<T, U, VEC> <<<blocks, threads, nshared, stream>>>(
                n1, n2, dY_data, X_data, mean, rstd, grad_gamma, grad_beta);
        } else {
            norms_device_kernel::cuApply_Rms_GammaBackward_Vec<T, U, VEC> <<<blocks, threads, nshared, stream>>>(
                n1, n2, dY_data, X_data, rstd, grad_gamma);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}


template<typename T, typename U, int VEC = 4>
void HostApply_LnRmsNormAddDrop_Gradient(T *grad_input, T *grad_gamma, T *grad_beta,
                                         const T *dY_data, const U *mean, const U *rstd,
                                         const T *X_data, T *temp_grad,
                                         int n1, int n2, const T *gamma, const T *beta,
                                         bool rms_only, bool normalized_affine,
                                         T *grad_residual, const uint8_t *drop_mask,
                                         U drop_pinv, int runing_type_enum) {

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int warp_size = at::cuda::warp_size();

    // compute grad_gamma(j) and grad_beta(j) and compute grad_input
    const dim3 block_dims_dx(512);
    const dim3 grid_dims_dx(n1);
    int nshared = (block_dims_dx.x / warp_size) * sizeof(U);

    if (runing_type_enum == 1) { // use norm only
        if (normalized_affine) {
            HostApply_LnRmsNorm_GammaBeta_Gradient<T, U, VEC>(grad_gamma, grad_beta, dY_data, mean, rstd, X_data, n1, n2, rms_only);
        }
        // compute grad input
        cuApply_LnRmsNorms_gradInput_Backward<T, U, VEC><<<grid_dims_dx, block_dims_dx, nshared, stream>>>(
            dY_data, X_data, mean, rstd, gamma, grad_input, n1, n2, rms_only);
    } else if (runing_type_enum == 2) {
        if (drop_mask != nullptr) {  // only add backward directly
            cuApply_AddDrops_Backward<T, U, VEC><<<grid_dims_dx, block_dims_dx, 0, stream>>>(
                dY_data, n1, n2, grad_input, grad_residual, drop_mask, drop_pinv);
        }
    } else {
        if (normalized_affine) {
            HostApply_LnRmsNorm_GammaBeta_Gradient<T, U, VEC>(grad_gamma, grad_beta, dY_data, mean, rstd, X_data, n1, n2, rms_only);
        }

        // compute grad input
        if (drop_mask != nullptr) {     // norms + add? + dropout
            cuApply_LnRmsNormsAddDrops_gradInput_Backward<T, U, VEC><<<grid_dims_dx, block_dims_dx, nshared, stream>>>(
                dY_data, X_data, mean, rstd, gamma,
                    n1, n2, rms_only, grad_input, grad_residual, temp_grad, drop_mask, drop_pinv);
        } else {    // norms + add
            cuApply_LnRmsNorms_gradInput_Backward<T, U, VEC><<<grid_dims_dx, block_dims_dx, nshared, stream>>>(
                dY_data, X_data, mean, rstd, gamma, grad_input, n1, n2, rms_only);
            // residual grad will compute outside
        }
    }
}

void cuda_ln_rms_norm_add_drops_gradient(int norm_type_enum, bool use_residual, bool use_dropout,
                                         const at::Tensor *dY, at::Tensor *mean, at::Tensor *rstd, const at::Tensor *X, at::Tensor *temp_grad,
                                         int n1, int n2,
                                         const at::Tensor *gamma, const at::Tensor *beta,
                                         at::Tensor *grad_input, at::Tensor *grad_gamma, at::Tensor *grad_beta, bool normalized_affine,
                                         at::Tensor *grad_residual, const at::Tensor *drop_mask, double drop_prob, bool is_training,
                                         int runing_type_enum) {
    bool use_norm = norm_type_enum > 0;
    bool rms_only = norm_type_enum == 2;
    bool use_drop = use_dropout && is_training && (drop_prob > 0);
    bool use_residual_drop = use_residual || use_drop;

    using namespace at;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        X->scalar_type(),
        "ln_rms_norm_add_drops_gradient_backward", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;

          if (n1 < block_size_VEC8) {
              HostApply_LnRmsNormAddDrop_Gradient<scalar_t, accscalar_t, 4>(
                  grad_input->data_ptr<scalar_t>(),
                  use_norm ? (normalized_affine ? grad_gamma->data_ptr<scalar_t>() : nullptr) : nullptr,
                  use_norm ? (rms_only ? nullptr : (normalized_affine ? grad_beta->data_ptr<scalar_t>() : nullptr)) : nullptr,
                  dY->data_ptr<scalar_t>(),
                  use_norm ? (rms_only ? nullptr : mean->data_ptr<accscalar_t>()) : nullptr,
                  use_norm ? (rstd->data_ptr<accscalar_t>()) : nullptr,
                  X->data_ptr<scalar_t>(),
                  (runing_type_enum == 0 && use_drop) ? (temp_grad->data_ptr<scalar_t>()) : nullptr,
                  n1, n2,
                  normalized_affine ? gamma->data_ptr<scalar_t>() : nullptr,
                  use_norm ? (rms_only ? nullptr : (normalized_affine ? beta->data_ptr<scalar_t>() : nullptr)) : nullptr,
                  rms_only,
                  normalized_affine,
                  use_residual ? grad_residual->data_ptr<scalar_t>() : nullptr,
                  use_drop ? drop_mask->data_ptr<uint8_t>() : nullptr,
                  static_cast<accscalar_t>(1 / (1 - drop_prob)),
                  runing_type_enum);
          } else {
              HostApply_LnRmsNormAddDrop_Gradient<scalar_t, accscalar_t, 8>(
                  grad_input->data_ptr<scalar_t>(),
                  use_norm ? (normalized_affine ? grad_gamma->data_ptr<scalar_t>() : nullptr) : nullptr,
                  use_norm ? (rms_only ? nullptr : (normalized_affine ? grad_beta->data_ptr<scalar_t>() : nullptr)) : nullptr,
                  dY->data_ptr<scalar_t>(),
                  use_norm ? (rms_only ? nullptr : mean->data_ptr<accscalar_t>()) : nullptr,
                  use_norm ? (rstd->data_ptr<accscalar_t>()) : nullptr,
                  X->data_ptr<scalar_t>(),
                  (runing_type_enum == 0 && use_drop) ? (temp_grad->data_ptr<scalar_t>()) : nullptr,
                  n1, n2,
                  normalized_affine ? gamma->data_ptr<scalar_t>() : nullptr,
                  use_norm ? (rms_only ? nullptr : (normalized_affine ? beta->data_ptr<scalar_t>() : nullptr)) : nullptr,
                  rms_only,
                  normalized_affine,
                  use_residual ? grad_residual->data_ptr<scalar_t>() : nullptr,
                  use_drop ? drop_mask->data_ptr<uint8_t>() : nullptr,
                  static_cast<accscalar_t>(1 / (1 - drop_prob)),
                  runing_type_enum);
          }
        }
    );
}

} // ln_rms_norms_add_drops

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 调用接口
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector <at::Tensor> lnRmsNorms_add_drops_forward(
    int norm_type_enum, bool use_residual, bool use_dropout,
    const at::Tensor &X, int normalized_shape,
    at::Tensor &weight, at::Tensor &bias, double epsilon, bool normalized_affine,
    const at::Tensor &residual, double drop_prob, bool is_training) {
    /** compute Norms( drop(x) + bias )
     * norm_type_enum: norm type  none/ln/rms
     * use_residual: none/residual
     * use_dropout: none/dropout
     * X: X tensor, x
     * normalized_shape: normalized shape for norm
     * weight: norm param
     * bias: norm param
     * epsilon: norm param
     * normalized_affine: norm has weight or bias
     * residual: bias tensor
     * drop_prob: drop rate of drop
     * is_training: drop running only training
     * */
    bool use_norm = norm_type_enum > 0;
    bool use_drop = use_dropout && is_training && (drop_prob > 0);
    bool use_residual_drop = use_residual || use_drop;

    int runing_type_enum;
    if (!use_norm && use_residual_drop) {
        runing_type_enum = 2;
    } else if (use_norm && !use_residual_drop) {
        runing_type_enum = 1;
    } else if (use_norm && use_residual_drop) {
        runing_type_enum = 0;
    } else {
        std::stringstream ss;
        ss << "using norm or using add drop\n";
        throw std::runtime_error(ss.str());
    }
    //////////////////////////////////////////// prepare norms ////////////////////////////////////////////
    // Compared to contiguous, it saves a reference count increment/decrement if *this is already contiguous,
    // at the cost in all cases of an extra pointer of stack usage, an extra branch to access, and an extra branch at destruction time.
    auto X_in = X.expect_contiguous();
    auto dst_options = X_in->options().requires_grad(false);
    at::Tensor output = torch::empty(X_in->sizes(), dst_options);
    at::Tensor rstd, mean, temp_add_output;

    int n1, n2;
    compute_n1_n2(*X_in, normalized_shape, n1, n2);

    if (use_norm) { // mean and sigma2 for norm
        auto acc_type = at::toAccumulateType(X_in->scalar_type(), /*is_cuda=*/true);

        rstd = at::empty({n1}, dst_options.dtype(acc_type));
        if (norm_type_enum == 1) {
            mean = at::empty({n1}, dst_options.dtype(acc_type));
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
    if (use_residual) {
        CHECK_INPUT(residual);
        TORCH_CHECK(X_in->sizes() == residual.sizes(), "X residual must has same shape");
    }

    auto mask_options = X_in->options().requires_grad(false).dtype(torch::kUInt8);
    torch::Tensor drop_mask;
    if (use_drop) {
        TORCH_CHECK(drop_prob >= 0 && drop_prob <= 1, "drop prob is during 0 and 1");
        drop_mask = at::empty_like(X, mask_options);
    }

    //////////////////////////////////////////// running function ////////////////////////////////////////////
    ln_rms_norms_add_drops::cuda_ln_rms_norm_add_drops(
        norm_type_enum, use_residual, use_dropout,
        &output, &mean, &rstd, X_in, &temp_add_output,
        n1, n2, &weight, &bias, epsilon, normalized_affine,
        &residual, &drop_mask, drop_prob, is_training, runing_type_enum);

    if (!use_drop) {
        drop_mask = torch::empty({1}, mask_options);
    }

    // if (!drop_mask.numel()) { ("drop_mask is nullptr\n"); } else { ("drop_mask is not nullptr\n"); }
    if (runing_type_enum == 0) {
        return {output, std::move(mean), std::move(rstd), std::move(temp_add_output), drop_mask};
    } else {
        return {output, std::move(mean), std::move(rstd), std::move(X), drop_mask};
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector <at::Tensor> lnRmsNorms_add_drops_backward(
    int norm_type_enum, bool use_residual, bool use_dropout,
    const at::Tensor &dY, at::Tensor &mean, at::Tensor &rstd, const at::Tensor &X, int normalized_shape,
    const at::Tensor &weight, const at::Tensor &bias, bool normalized_affine,
    const at::Tensor &drop_mask, double drop_prob, bool is_training) {

    bool use_norm = norm_type_enum > 0;
    bool use_drop = use_dropout && is_training && (drop_prob > 0);    // !!! use this after drop is correct
    bool use_residual_drop = use_residual || use_drop;

    int runing_type_enum;
    if (!use_norm && use_residual_drop) {
        runing_type_enum = 2;
    } else if (use_norm && !use_residual_drop) {
        runing_type_enum = 1;
    } else if (use_norm && use_residual_drop) {
        runing_type_enum = 0;
    } else {
        std::stringstream ss;
        ss << "using norm or using add drop\n";
        throw std::runtime_error(ss.str());
    }

    auto X_dout = dY.expect_contiguous();
    // check tensors
    CHECK_INPUT(dY);
    if (use_norm) {
        CHECK_INPUT(rstd);
    }
    if (use_drop) { CHECK_INPUT(drop_mask); }

    // prapare tensors
    at::Tensor grad_input = at::native::empty_like(*X_dout, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    at::Tensor grad_gamma, grad_beta, temp_grad;
    if (use_norm) {
        auto gamma = weight.expect_contiguous();
        auto beta = bias.expect_contiguous();

        grad_gamma = at::native::empty_like(
            *gamma,
            c10::nullopt /* dtype */,
            c10::nullopt /* layout */,
            c10::nullopt /* device */,
            c10::nullopt /* pin_memory */,
            LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        grad_beta = at::native::empty_like(
            *beta,
            c10::nullopt /* dtype */,
            c10::nullopt /* layout */,
            c10::nullopt /* device */,
            c10::nullopt /* pin_memory */,
            LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }

    if (runing_type_enum == 0 && use_drop) {
        temp_grad = at::native::empty_like(*X_dout, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }

    at::Tensor grad_residual;
    if (use_residual) {
        if (use_norm || use_drop) {
            grad_residual = at::empty_like(grad_input);
        } else { // if only compute add, return dY directly
            return {dY, std::move(grad_gamma), std::move(grad_beta), dY};
        }
    }

    int n1, n2;
    compute_n1_n2(*X_dout, normalized_shape, n1, n2);

    ln_rms_norms_add_drops::cuda_ln_rms_norm_add_drops_gradient(
        norm_type_enum, use_residual, use_dropout,
        &dY, &mean, &rstd, &X, &temp_grad,
        n1, n2, &weight, &bias,
        &grad_input, &grad_gamma, &grad_beta, normalized_affine,
        &grad_residual, &drop_mask, drop_prob, is_training, runing_type_enum);

    if (runing_type_enum == 0 && !use_drop) {   // norm + add
        return {grad_input, std::move(grad_gamma), std::move(grad_beta), grad_input};
    }

    return {std::move(grad_input), std::move(grad_gamma), std::move(grad_beta), std::move(grad_residual)};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FusedLnRmsNormAddDropFunction : public torch::autograd::Function<FusedLnRmsNormAddDropFunction> {
public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext *ctx,
      const torch::autograd::Variable &x,
      torch::autograd::Variable &weight,
      torch::autograd::Variable &bias,
      int normalized_shape,
      double normalized_eps,
      bool normalized_affine,
      const torch::autograd::Variable &residual,
      double drop_prob,
      int norm_type_enum, bool use_residual, bool use_dropout,
      bool training) {

      ctx->saved_data["norm_type_enum"] = norm_type_enum;
      ctx->saved_data["use_residual"] = use_residual;
      ctx->saved_data["use_dropout"] = use_dropout;
      ctx->saved_data["normalized_shape"] = normalized_shape;
      ctx->saved_data["normalized_affine"] = normalized_affine;
      ctx->saved_data["drop_prob"] = drop_prob;
      ctx->saved_data["training"] = training;

      // ret: output, mean, rstd, temp_add_output, drop_mask
      // ctx.save_for_backward(weight, bias, mean, rstd, temp_add_output, drop_mask)
      auto ret = lnRmsNorms_add_drops_forward(norm_type_enum, use_residual, use_dropout,
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
      bool use_residual = ctx->saved_data["use_residual"].toBool();
      bool use_dropout = ctx->saved_data["use_dropout"].toBool();
      int normalized_shape = ctx->saved_data["normalized_shape"].toInt();
      bool normalized_affine = ctx->saved_data["normalized_affine"].toBool();
      double drop_prob = ctx->saved_data["drop_prob"].toDouble();
      bool training = ctx->saved_data["training"].toBool();

      // weight, bias, mean, rstd, temp_add_output, drop_mask = ctx.saved_tensors
      // ret: grad_input, grad_weight, grad_bias, grad_residual
      // ctx->save_for_backward({weight, bias, ret[1], ret[2], ret[3], ret[4]});
      auto ret = lnRmsNorms_add_drops_backward(norm_type_enum, use_residual, use_dropout,
                                               gradOutput[0], saved[2], saved[3], saved[4], normalized_shape,
                                               saved[0], saved[1], normalized_affine, saved[5], drop_prob, training);
      return {ret[0], ret[1], ret[2],
              torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable(),
              ret[3],
              torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable()};
  }
};

std::tuple <Tensor, Tensor> fuse_lnrmsnorms_add_drop_forward_autograd(const Tensor &x, Tensor &weight, Tensor &bias,
                                                                      int normalized_shape, double normalized_eps, bool normalized_affine,
                                                                      const Tensor &residual,
                                                                      double drop_prob,
                                                                      int norm_type_enum, bool use_residual, bool use_dropout,
                                                                      bool training) {
    auto result = FusedLnRmsNormAddDropFunction::apply(x, weight, bias, normalized_shape, normalized_eps, normalized_affine,
                                                       residual, drop_prob, norm_type_enum, use_residual, use_dropout, training);
    return std::tuple<Tensor, Tensor>(result[0], result[1]);
}

} // native
} // at
