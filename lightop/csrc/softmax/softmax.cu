#pragma once

#include <assert.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <type_traits>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <c10/macros/Macros.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/NumericLimits.cuh>

#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/DeviceUtils.cuh>

#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/Dispatch.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Half.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/native/Resize.h>

#include <ATen/Config.h>
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/ATenHIPGeneral.h>
#include <ATen/hip/HIPDevice.h>
#include <ATen/hip/HIPContext.h>

#include <ATen/CUDAFunctions.h>
#include "cuda_meta.h"

using namespace std;

#if defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#else
#include <cuda_fp16.h>
#include <cuda.h>
#endif

#include "softmax.h"
#include "softmax_math.cuh"
#include "softmax_lastdim_impl.cuh"
#include "softmax_spatial_impl.cuh"


namespace at {
namespace native {

namespace softmax_kernel {

template<bool is_log_softmax>
void fuse_softmax_lastdim_fwd(const Tensor &input, const Tensor& output, const bool &half_to_float,
                              const int64_t &outer_size, const int64_t &dim_size, const int64_t &inner_size,
                              cudaStream_t stream) {
    // This kernel spawns a block per each element in the batch.
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "fuse_softmax_lastdim_fwd", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        if (!half_to_float) {
            int half_threshold = 2048;
            int float_threshold = 4096;

            if ((input.scalar_type() == ScalarType::Float && dim_size <= float_threshold) || 
                (input.scalar_type() == ScalarType::Half && dim_size <= half_threshold)) {
                auto output_ptr = output.data_ptr<scalar_t>();
                auto input_ptr = input.data_ptr<scalar_t>();
                softmax_lastdim::softmax_lastdim_smallsize_forward<scalar_t, scalar_t, accscalar_t, is_log_softmax>(
                    output_ptr, input_ptr, dim_size, outer_size, stream);
            } else {
                auto output_ptr = output.data_ptr<scalar_t>();
                auto input_ptr = input.data_ptr<scalar_t>();
                softmax_lastdim::softmax_lastdim_largesize_forward<scalar_t, scalar_t, accscalar_t, is_log_softmax>(
                    output_ptr, input_ptr, dim_size, outer_size, stream);
            }
        } else {
            if (dim_size <= 2048) {
                auto output_ptr = output.data_ptr<accscalar_t>();
                auto input_ptr = input.data_ptr<scalar_t>();
                softmax_lastdim::softmax_lastdim_smallsize_forward<scalar_t, accscalar_t, accscalar_t, is_log_softmax>(
                    output_ptr, input_ptr, dim_size, outer_size, stream);
            } else {
                auto output_ptr = output.data_ptr<accscalar_t>();
                auto input_ptr = input.data_ptr<scalar_t>();
                softmax_lastdim::softmax_lastdim_largesize_forward<scalar_t, accscalar_t, accscalar_t, is_log_softmax>(
                    output_ptr, input_ptr, dim_size, outer_size, stream);
            }
        }
    });
}

template<bool is_log_softmax>
void fuse_softmax_spatial_fwd(const Tensor &input, const Tensor& output, const bool &half_to_float,
                              const int64_t &outer_size, const int64_t &dim_size, const int64_t &inner_size,
                              const cudaStream_t &stream) {
    // This kernel spawns a block per each element in the batch.
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "fuse_softmax_spatial_fwd", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        if (!half_to_float) {
            auto output_ptr = output.data_ptr<scalar_t>();
            auto input_ptr = input.data_ptr<scalar_t>();
            softmax_spatial::softmax_spatial_forward<scalar_t, scalar_t, accscalar_t, is_log_softmax>(
                output_ptr, input_ptr, outer_size, dim_size, inner_size, stream);

        } else {
            auto output_ptr = output.data_ptr<accscalar_t>();
            auto input_ptr = input.data_ptr<scalar_t>();
            softmax_spatial::softmax_spatial_forward<scalar_t, accscalar_t, accscalar_t, is_log_softmax>(
                output_ptr, input_ptr, outer_size, dim_size, inner_size, stream);  
        }
    });
}

/* 启用kernel的函数 */
template<bool is_log_softmax>
Tensor fuse_softmax_fwd(const Tensor &input_, const int64_t dim_, const bool half_to_float, const Tensor& output) {
    if (half_to_float) {
        TORCH_CHECK(input_.scalar_type() == ScalarType::Half, "conversion is supported for Half type only");
    }
    auto input = input_.contiguous();
    if (input.numel() == 0) { return output; }

    static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_t for half should be float");
    if (input.dim() == 0) input = input.view(1);
    int64_t dim = maybe_wrap_dim(dim_, input.dim());
    TORCH_CHECK(dim >=0 && dim < input.dim(), "dim must be non-negative and less than input dimensions");

    int64_t outer_size = 1;
    int64_t dim_size = input.size(dim);
    int64_t inner_size = 1;
    for (int64_t i = 0; i < dim; ++i) { outer_size *= input.size(i); }
    for (int64_t i = dim + 1; i < input.dim(); ++i) { inner_size *= input.size(i); }
    // printf("outer_size=%d, dim_size=%d, inner_size=%d, input.ndim=%d, \n", outer_size, dim_size, inner_size, input.dim());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // This kernel spawns a block per each element in the batch.
    if (inner_size == 1) {
        fuse_softmax_lastdim_fwd<is_log_softmax>(input, output, half_to_float,
                                                 outer_size, dim_size, inner_size, stream);
        // This kernel runs in a 2D grid, where each application along y dimension has a fixed
        // outer_size, and runs in parallel over inner_size. Dimension x is parallel over outer_size.
        // Reductions over dim are done in a single-threaded manner.
        return output;
    } else {
        if (softmax_op::softmax_spatial_fwd_use_transpose(outer_size, dim_size, inner_size)) {
            // 通过输入的维度转换，将需要计算softmax的维度调到最后一维
            auto input_transpose = input.transpose(dim, input.dim()-1).contiguous();
            auto output_transpose = output.transpose(dim, input.dim()-1).contiguous();
            fuse_softmax_lastdim_fwd<is_log_softmax>(input_transpose, output_transpose, half_to_float,
                                                     outer_size*inner_size, dim_size, 1, stream); 
            // 输出的维度再调换一次
            return output_transpose.transpose(dim, input.dim()-1).contiguous();
        } else {
            fuse_softmax_spatial_fwd<is_log_softmax>(input, output, half_to_float,
                                                     outer_size, dim_size, inner_size, stream);
            return output;
        }
    }
}

template<bool is_log_softmax>
void fuse_softmax_lastdim_bwd(const Tensor &grad_input, const Tensor &grad, const Tensor& output, const bool &half_to_float,
                              const int64_t &outer_size, const int64_t &dim_size, const int64_t &inner_size,
                              cudaStream_t stream) {
    // This kernel spawns a block per each element in the batch.
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_input.scalar_type(), "fuse_softmax_lastdim_bwd", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        if (!half_to_float) {
            int half_threshold = 1024;
            int float_threshold = 1024;

            if ((grad_input.scalar_type() == ScalarType::Float && dim_size <= float_threshold) || 
                (grad_input.scalar_type() == ScalarType::Half && dim_size <= half_threshold)) {
                auto grad_input_ptr = grad_input.data_ptr<scalar_t>();
                auto grad_ptr = grad.data_ptr<scalar_t>();
                auto output_ptr = output.data_ptr<scalar_t>();
                softmax_lastdim::softmax_lastdim_smallsize_backward<scalar_t, scalar_t, accscalar_t, is_log_softmax>(
                    grad_input_ptr, grad_ptr, output_ptr, dim_size, outer_size, stream);
            } else {
                auto grad_input_ptr = grad_input.data_ptr<scalar_t>();
                auto grad_ptr = grad.data_ptr<scalar_t>();
                auto output_ptr = output.data_ptr<scalar_t>();
                softmax_lastdim::softmax_lastdim_largesize_backward<scalar_t, scalar_t, accscalar_t, is_log_softmax>(
                    grad_input_ptr, grad_ptr, output_ptr, dim_size, outer_size, stream);
            }
        } else {
            if (dim_size <= 2048) {
                auto grad_input_ptr = grad_input.data_ptr<scalar_t>();
                auto grad_ptr = grad.data_ptr<accscalar_t>();
                auto output_ptr = output.data_ptr<accscalar_t>();
                softmax_lastdim::softmax_lastdim_smallsize_backward<scalar_t, accscalar_t, accscalar_t, is_log_softmax>(
                    grad_input_ptr, grad_ptr, output_ptr, dim_size, outer_size, stream);
            } else {
                auto grad_input_ptr = grad_input.data_ptr<scalar_t>();
                auto grad_ptr = grad.data_ptr<accscalar_t>();
                auto output_ptr = output.data_ptr<accscalar_t>();
                softmax_lastdim::softmax_lastdim_largesize_backward<scalar_t, accscalar_t, accscalar_t, is_log_softmax>(
                    grad_input_ptr, grad_ptr, output_ptr, dim_size, outer_size, stream);
            }
        }
    });
}

template<bool is_log_softmax>
void fuse_softmax_spatial_bwd(const Tensor &grad_input, const Tensor &grad, const Tensor& output, const bool &half_to_float,
                              const int64_t &outer_size, const int64_t &dim_size, const int64_t &inner_size,
                              cudaStream_t stream) {
    uint32_t smem_size;
    dim3 grid, block;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_input.scalar_type(), "fuse_softmax_spatial_bwd", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        if (!half_to_float) {
            auto grad_input_ptr = grad_input.data_ptr<scalar_t>();
            auto grad_ptr = grad.data_ptr<scalar_t>();
            auto output_ptr = output.data_ptr<scalar_t>();
            softmax_spatial::softmax_spatial_backward<scalar_t, scalar_t, accscalar_t, is_log_softmax>(
                grad_input_ptr, grad_ptr, output_ptr, outer_size, dim_size, inner_size, stream);
        } else {
            auto grad_input_ptr = grad_input.data_ptr<scalar_t>();
            auto grad_ptr = grad.data_ptr<accscalar_t>();
            auto output_ptr = output.data_ptr<accscalar_t>();
            softmax_spatial::softmax_spatial_backward<scalar_t, accscalar_t, accscalar_t, is_log_softmax>(
                grad_input_ptr, grad_ptr, output_ptr, outer_size, dim_size, inner_size, stream);
        }
    });
}

template<bool is_log_softmax>
Tensor fuse_softmax_bwd(const Tensor &grad_, const Tensor &output_, int64_t dim_, bool half_to_float, const Tensor &grad_input) {
    int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
    if (grad_.numel() == 0) { return grad_input; }

    auto grad = grad_.contiguous();
    static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_t for half should be float");
    if (grad.dim() == 0) grad = grad.view(1);
    TORCH_CHECK(dim >=0 && dim < grad.dim(), "dim must be non-negative and less than input dimensions");
    auto output = output_.contiguous();
    if (output.dim() == 0) output = output.view(1);

    int64_t outer_size = 1;
    int64_t dim_size = output.size(dim);
    int64_t inner_size = 1;
    for (int64_t i = 0; i < dim; ++i) { outer_size *= output.size(i); }
    for (int64_t i = dim + 1; i < output.dim(); ++i) { inner_size *= output.size(i); }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int ndim = output.dim();

    // See descriptions of kernels above.
    if (inner_size == 1) {
        fuse_softmax_lastdim_bwd<is_log_softmax>(grad_input, grad, output, half_to_float,
                                                 outer_size, dim_size, inner_size, stream);
        return grad_input;
    } else {
        if (softmax_op::softmax_spatial_bwd_use_transpose(outer_size, dim_size, inner_size)) {
            // 通过输入的维度转换，将需要计算softmax的维度调到最后一维
            auto grad_input_transpose = grad_input.transpose(dim, ndim-1).contiguous();
            auto grad_transpose = grad.transpose(dim, ndim-1).contiguous();
            auto output_transpose = output.transpose(dim, ndim-1).contiguous();
            fuse_softmax_lastdim_bwd<is_log_softmax>(grad_input_transpose, grad_transpose, output_transpose, half_to_float,
                                                     outer_size*inner_size, dim_size, 1, stream); 
            // 输出的维度再调换一次
            return grad_input_transpose.transpose(dim, ndim-1).contiguous();
        } else {
            fuse_softmax_spatial_bwd<is_log_softmax>(grad_input, grad, output, half_to_float,
                                                     outer_size, dim_size, inner_size, stream);
            return grad_input;
        }

        return grad_input;
    }
}

}   // namespace softmax_kernel


namespace {     // 调用softmax kernel的函数

struct TORCH_API structured__fused_softmax : public at::impl::MetaBase {  
    void meta(const Tensor& input, const int64_t dim, const bool half_to_float) {
        int64_t dim_ = maybe_wrap_dim(dim, input.dim());

        auto output_options = input.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

        if (half_to_float) {
            output_options = output_options.dtype(ScalarType::Float);
        }

        int64_t input_dim = input.dim() > 0 ? input.dim() : 1; 
        TORCH_CHECK( dim_ >= 0 && dim_ < input_dim,
            "dim must be non-negative and less than input dimensions");

        #ifdef TORCH_VERSION_OVER_110
	        set_output_raw_strided(0,input.sizes(),{}, output_options);
	    #else
	        set_output(input.sizes(), output_options);
	    #endif
    }
    void impl(const at::Tensor &input, int64_t dim, bool half_to_float, at::Tensor &out) {
        out = softmax_kernel::fuse_softmax_fwd<false>(input, dim, half_to_float, out);
    }
    
};
TORCH_META_FUNC_FINAL(_fused_softmax)

struct TORCH_API structured__fused_softmax_backward : public at::impl::MetaBase {  
    void meta(const Tensor& grad, const Tensor& output, int64_t dim, const Tensor& input) {
        TensorArg grad_arg{grad, "grad", 1}, output_arg{output, "output", 2};
        checkSameSize("softmax_backward", grad_arg, output_arg);

        int64_t dim_ = maybe_wrap_dim(dim, grad.dim());

        auto grad_input_options =
            grad.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

        bool half_to_float = grad.scalar_type() != input.scalar_type();
        if (half_to_float) {
            // The code below is only valid for the CUDA implementation. It's "okay"
            // to put it here because half-to-float conversion is not supported by
            // the CPU implementation of _softmax. There is a TORCH_CHECK in the CUDA
            // implementation that should ideally go here as well, but there is at least
            // one test in which the grad and input dtypes do not match for the CPU
            // implementation of this kernel and it is not true that the grad type is
            // float and the input dtype is half (see #63057).
            if (grad.scalar_type() == ScalarType::Float &&
                input.scalar_type() == ScalarType::Half) {
            grad_input_options = grad_input_options.dtype(ScalarType::Half);
            }
        }

        int64_t grad_dim = grad.dim() > 0 ? grad.dim() : 1;
        TORCH_CHECK(dim_ >= 0 && dim_ < grad_dim,
            "dim must be non-negative and less than input dimensions");

        #ifdef TORCH_VERSION_OVER_110
            set_output_raw_strided(0,grad.sizes(),{}, grad_input_options);
        #else
            set_output(grad.sizes(), grad_input_options);
        #endif
    }
    void impl(const at::Tensor &grad, const at::Tensor &output, int64_t dim, bool half_to_float, at::Tensor &grad_input) {
        grad_input = softmax_kernel::fuse_softmax_bwd<false>(grad, output, dim, half_to_float, grad_input);          
    }
};
TORCH_META_FUNC_FINAL(_fused_softmax_backward)

struct TORCH_API structured__fused_log_softmax_cuda_out : public at::impl::MetaBase {  
    void meta(const Tensor& input, const int64_t dim, const bool half_to_float) {
        int64_t dim_ = maybe_wrap_dim(dim, input.dim());

        auto output_options =
            input.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

        if (half_to_float) {
            output_options = output_options.dtype(ScalarType::Float);
        }

        int64_t input_dim = input.dim() > 0 ? input.dim() : 1; 
        TORCH_CHECK(dim_ >= 0 && dim_ < input_dim,
            "dim must be non-negative and less than input dimensions");

        #ifdef TORCH_VERSION_OVER_110
	        set_output_raw_strided(0,input.sizes(),{}, output_options);
	    #else
	        set_output(input.sizes(), output_options);
	    #endif
    }
    void impl(const at::Tensor &input, int64_t dim, bool half_to_float, at::Tensor &out) {
        out = softmax_kernel::fuse_softmax_fwd<true>(input, dim, half_to_float, out);
    }
};
TORCH_META_FUNC_FINAL(_fused_log_softmax_cuda_out)

struct TORCH_API structured__fused_log_softmax_backward_cuda_out : public at::impl::MetaBase {
    void meta(const Tensor& grad, const Tensor& output, int64_t dim, const Tensor& input) {
        TensorArg grad_arg{grad, "grad", 1}, output_arg{output, "output", 2};
        checkSameSize("softmax_backward", grad_arg, output_arg);

        int64_t dim_ = maybe_wrap_dim(dim, grad.dim());

        auto grad_input_options =
            grad.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

        bool half_to_float = grad.scalar_type() != input.scalar_type();
        if (half_to_float) {
            // The code below is only valid for the CUDA implementation. It's "okay"
            // to put it here because half-to-float conversion is not supported by
            // the CPU implementation of _softmax. There is a TORCH_CHECK in the CUDA
            // implementation that should ideally go here as well, but there is at least
            // one test in which the grad and input dtypes do not match for the CPU
            // implementation of this kernel and it is not true that the grad type is
            // float and the input dtype is half (see #63057).
            if (grad.scalar_type() == ScalarType::Float &&
                input.scalar_type() == ScalarType::Half) {
            grad_input_options = grad_input_options.dtype(ScalarType::Half);
            }
        }

        int64_t grad_dim = grad.dim() > 0 ? grad.dim() : 1;
        TORCH_CHECK(dim_ >= 0 && dim_ < grad_dim,
            "dim must be non-negative and less than input dimensions");

        #ifdef TORCH_VERSION_OVER_110
            set_output_raw_strided(0,grad.sizes(),{}, grad_input_options);
        #else
            set_output(grad.sizes(), grad_input_options);
        #endif
    }
    void impl(const at::Tensor &grad, const at::Tensor &output, int64_t dim, bool half_to_float, at::Tensor &grad_input) {
        grad_input = softmax_kernel::fuse_softmax_bwd<true>(grad, output, dim, half_to_float, grad_input);          
    }
};
TORCH_META_FUNC_FINAL(_fused_log_softmax_backward_cuda_out)
}   // namespace


//// 启用softmax的torch对外接口
Tensor fuse_softmax_forward(const Tensor &input, const int64_t &dim, const bool &half_to_float) {
    c10::optional<Device> common_device = c10::nullopt;
    (void)common_device; // Suppress unused variable warning

    bool half_to_float_new = half_to_float && input.scalar_type() == ScalarType::Half;

    c10::impl::check_and_update_common_device(common_device, input, "fuse_softmax_forward", "input");
    structured__fused_softmax_functional op;
    op.meta(input, dim, half_to_float_new);
    op.impl(input, dim, half_to_float_new, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor fuse_softmax_backward(const Tensor& grad, const Tensor& output, 
                             const int64_t &dim, const Tensor& input) {
    bool half_to_float = grad.scalar_type() != input.scalar_type();
    if (half_to_float) {
    TORCH_CHECK(
        (grad.scalar_type() == ScalarType::Float &&
         input.scalar_type() == ScalarType::Half),
        "expected input and grad types to match, or input to be at::Half and grad to be at::Float");
    }
    c10::optional<Device> common_device = c10::nullopt;
    (void)common_device; // Suppress unused variable warning

    c10::impl::check_and_update_common_device(common_device, grad, "wrapper__softmax_backward_data", "grad");
    c10::impl::check_and_update_common_device(common_device, output, "wrapper__softmax_backward_data", "output");
    c10::impl::check_and_update_common_device(common_device, input, "wrapper__softmax_backward_data", "input");
    structured__fused_softmax_backward_functional op;
    op.meta(grad, output, dim, input);
    Tensor tmp = grad * output;
    op.impl(tmp, output, dim, half_to_float, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor fuse_log_softmax_forward(const at::Tensor &input, const int64_t &dim, 
                                const bool &half_to_float) {
    c10::optional<Device> common_device = c10::nullopt;
    (void)common_device; // Suppress unused variable warning

    bool half_to_float_new = half_to_float && input.scalar_type() == ScalarType::Half;

    c10::impl::check_and_update_common_device(common_device, input, "wrapper__log_softmax", "input");
    structured__fused_log_softmax_cuda_out_functional op;
    op.meta(input, dim, half_to_float_new);
    op.impl(input, dim, half_to_float_new, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

Tensor fuse_log_softmax_backward(const at::Tensor &grad, const at::Tensor &output, 
                                 const int64_t &dim, const at::Tensor &input) {
    bool half_to_float = grad.scalar_type() != input.scalar_type();
    if (half_to_float) {
    TORCH_CHECK(
        (grad.scalar_type() == ScalarType::Float &&
         input.scalar_type() == ScalarType::Half),
        "expected input and grad types to match, or input to be at::Half and grad to be at::Float");
    }
    c10::optional<Device> common_device = c10::nullopt;
    (void)common_device; // Suppress unused variable warning

    c10::impl::check_and_update_common_device(common_device, grad, "wrapper__log_softmax_backward_data", "grad");
    c10::impl::check_and_update_common_device(common_device, output, "wrapper__log_softmax_backward_data", "output");
    c10::impl::check_and_update_common_device(common_device, input, "wrapper__log_softmax_backward_data", "input");
    structured__fused_log_softmax_backward_cuda_out_functional op;
    op.meta(grad, output, dim, input);
    op.impl(grad, output, dim, half_to_float, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}

}   // namespace native
}   // namespace at
