/*
  Implementation is based on the algorihtm presented in pages 1685-1686 of:

  DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems
    52(4):1679-1696, August 2016
    doi: 10.1109/TAES.2016.140952
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <c10/macros/Macros.h>
#include <grid_sampler.cuh>

#include <limits>
#include <cmath>

namespace at {
namespace native {

  using namespace at::cuda::detail;

  using at::native::detail::GridSamplerInterpolation;
  using at::native::detail::GridSamplerPadding;


template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_2d_kernel(
      const index_t nthreads,
      TensorInfo<const scalar_t, index_t> input,
      TensorInfo<const scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> output,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners) {

    using opmath_t = at::opmath_type<scalar_t>;
    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];
    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];
    index_t out_sN = output.strides[0];
    index_t out_sC = output.strides[1];
    index_t out_sH = output.strides[2];
    index_t out_sW = output.strides[3];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t n = index / (out_H * out_W);
      const index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y coordinates from grid
      opmath_t x = grid.data[grid_offset];
      opmath_t y = grid.data[grid_offset + grid_sCoor];

      opmath_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
      opmath_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        index_t ix_nw = static_cast<index_t>(::floor(ix));
        index_t iy_nw = static_cast<index_t>(::floor(iy));
        index_t ix_ne = ix_nw + 1;
        index_t iy_ne = iy_nw;
        index_t ix_sw = ix_nw;
        index_t iy_sw = iy_nw + 1;
        index_t ix_se = ix_nw + 1;
        index_t iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        opmath_t nw = (ix_se - ix)    * (iy_se - iy);
        opmath_t ne = (ix    - ix_sw) * (iy_sw - iy);
        opmath_t sw = (ix_ne - ix)    * (iy    - iy_ne);
        opmath_t se = (ix    - ix_nw) * (iy    - iy_nw);

        // calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          opmath_t out_acc = 0;
          if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
          }
          if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
          }
          if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
          }
          if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
          }
          *out_ptr_NCHW = out_acc;
        }
      } 
      // else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      //   index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
      //   index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));

      //   // assign nearest neighbour pixel value to output pixel
      //   auto inp_ptr_NC = input.data + n * inp_sN;
      //   auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
      //   for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
      //     if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
      //       *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
      //     } else {
      //       *out_ptr_NCHW = static_cast<scalar_t>(0);
      //     }
      //   }
      // } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {

      //   ix = grid_sampler_unnormalize(x, inp_W, align_corners);
      //   iy = grid_sampler_unnormalize(y, inp_H, align_corners);

      //   opmath_t ix_nw = std::floor(ix);
      //   opmath_t iy_nw = std::floor(iy);

      //   const opmath_t tx = ix - ix_nw;
      //   const opmath_t ty = iy - iy_nw;

      //   auto inp_ptr_NC = input.data + n * inp_sN;
      //   auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
      //   for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
      //     opmath_t coefficients[4];

      //     #pragma unroll 4
      //     for (index_t i = 0; i < 4; ++i) {
      //       coefficients[i] = cubic_interp1d(
      //         get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
      //         get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
      //         get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
      //         get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
      //         tx);
      //     }

      //     *out_ptr_NCHW = cubic_interp1d(
      //       coefficients[0],
      //       coefficients[1],
      //       coefficients[2],
      //       coefficients[3],
      //       ty);
      //   }
      // }
    }
  }

// Note [Passing pointer and offset to fastAtomicAdd]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// For its internal bounds checking, fastAtomicAdd needs to know where the destination address
// lies relative to the entire tensor, so we pass the base grad_input.data and full offset information,
// including batch * channel offset (NC_offset).

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void grid_sampler_2d_backward_kernel(
    const index_t nthreads,
    TensorInfo<const scalar_t, index_t> grad_output,
    TensorInfo<const scalar_t, index_t> input,
    TensorInfo<const scalar_t, index_t> grid,
    TensorInfo<scalar_t, index_t> grad_input,  // initialized to zeros (or unused if input_requires_grad is false)
    TensorInfo<scalar_t, index_t> grad_grid,   // initialized to empty
    const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    const index_t grad_input_memory_span,
    const bool input_requires_grad) {

    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];
    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];
    index_t gOut_sN = grad_output.strides[0];
    index_t gOut_sC = grad_output.strides[1];
    index_t gOut_sH = grad_output.strides[2];
    index_t gOut_sW = grad_output.strides[3];
    // gInp_* (and NC_offset below) are not really needed if input_requires_grad is false.
    index_t gInp_sN;
    index_t gInp_sC;
    index_t gInp_sH;
    index_t gInp_sW;
    if (input_requires_grad) {
      gInp_sN = grad_input.strides[0];
      gInp_sC = grad_input.strides[1];
      gInp_sH = grad_input.strides[2];
      gInp_sW = grad_input.strides[3];
    }
    index_t gGrid_sW = grad_grid.strides[2];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t n = index / (out_H * out_W);
      const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y coordinates from grid
      scalar_t x = grid.data[grid_offset];
      scalar_t y = grid.data[grid_offset + grid_sCoor];

      // multipliers for gradients on ix and iy
      scalar_t gix_mult, giy_mult;
      scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult);
      scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &giy_mult);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        index_t ix_nw = static_cast<index_t>(std::floor(ix));
        index_t iy_nw = static_cast<index_t>(std::floor(iy));
        index_t ix_ne = ix_nw + 1;
        index_t iy_ne = iy_nw;
        index_t ix_sw = ix_nw;
        index_t iy_sw = iy_nw + 1;
        index_t ix_se = ix_nw + 1;
        index_t iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        scalar_t nw = (ix_se - ix)    * (iy_se - iy);
        scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
        scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
        scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
        const scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        index_t NC_offset = n * gInp_sN;
        const scalar_t *inp_ptr_NC = input.data + n * inp_sN;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
          const scalar_t gOut = *gOut_ptr_NCHW;

          if (input_requires_grad) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            safe_add_2d(grad_input.data, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut, NC_offset, grad_input_memory_span);
            safe_add_2d(grad_input.data, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut, NC_offset, grad_input_memory_span);
            safe_add_2d(grad_input.data, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut, NC_offset, grad_input_memory_span);
            safe_add_2d(grad_input.data, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut, NC_offset, grad_input_memory_span);
          }

          // calculate grad_grid
          if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
            gix -= nw_val * (iy_se - iy) * gOut;
            giy -= nw_val * (ix_se - ix) * gOut;
          }
          if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
            gix += ne_val * (iy_sw - iy) * gOut;
            giy -= ne_val * (ix - ix_sw) * gOut;
          }
          if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
            gix -= sw_val * (iy - iy_ne) * gOut;
            giy += sw_val * (ix_ne - ix) * gOut;
          }
          if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
            gix += se_val * (iy - iy_nw) * gOut;
            giy += se_val * (ix - ix_nw) * gOut;
          }
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = gix_mult * gix;
        gGrid_ptr_NHW[1] = giy_mult * giy;
      } 
      // else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      //   if (input_requires_grad) {
      //     index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
      //     index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));

      //     // assign nearest neighbour pixel value to output pixel
      //     const scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
      //     index_t NC_offset = n * gInp_sN;
      //     for (index_t c = 0; c < C; ++c, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
      //       // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
      //       safe_add_2d(grad_input.data, iy_nearest, ix_nearest, gInp_sH, gInp_sW, inp_H, inp_W, *gOut_ptr_NCHW, NC_offset, grad_input_memory_span);
      //     }
      //   }

      //   // assuming grad_grid is contiguous
      //   // thus we can
      //   //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
      //   //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
      //   scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
      //   gGrid_ptr_NHW[0] = static_cast<scalar_t>(0);
      //   gGrid_ptr_NHW[1] = static_cast<scalar_t>(0);
      // } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {

      //   ix = grid_sampler_unnormalize_set_grad(x, inp_W, align_corners, &gix_mult);
      //   iy = grid_sampler_unnormalize_set_grad(y, inp_H, align_corners, &giy_mult);

      //   scalar_t ix_nw = std::floor(ix);
      //   scalar_t iy_nw = std::floor(iy);

      //   const scalar_t tx = ix - ix_nw;
      //   const scalar_t ty = iy - iy_nw;

      //   scalar_t x_coeffs[4];
      //   scalar_t y_coeffs[4];
      //   scalar_t x_coeffs_grad[4];
      //   scalar_t y_coeffs_grad[4];

      //   get_cubic_upsampling_coefficients<scalar_t>(x_coeffs, tx);
      //   get_cubic_upsampling_coefficients<scalar_t>(y_coeffs, ty);
      //   get_cubic_coefficients_grad<scalar_t>(x_coeffs_grad, tx);
      //   get_cubic_coefficients_grad<scalar_t>(y_coeffs_grad, ty);

      //   scalar_t gix = static_cast<scalar_t>(0);
      //   scalar_t giy = static_cast<scalar_t>(0);

      //   const scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
      //   index_t NC_offset = n * gInp_sN;
      //   const scalar_t *inp_ptr_NC = input.data + n * inp_sN;

      //   for (index_t c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, NC_offset += gInp_sC, inp_ptr_NC+= inp_sC) {
      //     const scalar_t gOut = *gOut_ptr_NCHW;

      //     #pragma unroll 4
      //     for (index_t i = 0; i < 4; ++i) {
      //       #pragma unroll 4
      //       for (index_t j = 0; j < 4; ++j) {

      //         if (input_requires_grad) {
      //           // set input gradient. See Note [Passing pointer and offset to fastAtomicAdd].
      //           add_value_bounded<scalar_t>(grad_input.data, ix_nw - 1 + i, iy_nw - 1 + j, inp_W, inp_H, gInp_sW, gInp_sH,
      //             gOut * x_coeffs[i] * y_coeffs[j],
      //             padding_mode,
      //             align_corners,
      //             NC_offset,
      //             grad_input_memory_span);
      //         }

      //         // set grid gradient
      //         scalar_t val = get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw - 1 + i, iy_nw - 1 + j,
      //           inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners);

      //         gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gOut;
      //         giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gOut;
      //       }
      //     }
      //   }

      //   scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
      //   gGrid_ptr_NHW[0] = gix_mult * gix;
      //   gGrid_ptr_NHW[1] = giy_mult * giy;
      // }
    }
  }


  bool canUse32BitIndexMath(const TensorBase& t, int64_t max_elem) {
    auto elements = t.sym_numel();
    if (elements >= max_elem) {
      return false;
    }
    if (elements == 0) {
      return max_elem > 0;
    }
  
    c10::SymInt offset = 0;
    auto linearId = elements - 1;
  
    // NOTE: Assumes all strides are positive, which is true for now
    for (auto i = t.dim() - 1; i >= 0; --i) {
      auto curDimIndex = linearId % t.sym_size(i);
      auto curDimOffset = curDimIndex * t.sym_stride(i);
      offset += curDimOffset;
      linearId /= t.sym_size(i);
    }
  
    if (offset >= max_elem) {
      return false;
    }
  
    return true;
  }

  void launch_grid_sampler_2d_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  check_grid_sampler_common(input, grid);
  check_grid_sampler_2d(input, grid);
  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  int64_t count = N * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      input.scalar_type(), "grid_sampler_2d_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(output)) {
        grid_sampler_2d_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<const scalar_t, int>(input),
            getTensorInfo<const scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        grid_sampler_2d_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<const scalar_t, int64_t>(input),
            getTensorInfo<const scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}

  void launch_grid_sampler_2d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool,2> output_mask) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  check_grid_sampler_common(input, grid);
  check_grid_sampler_2d(input, grid);
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("grid_sampler_2d_backward_cuda");
  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);

  // If `input` gradient is not required, we skip computing it -- not needing to create
  // the tensor to hold the gradient can markedly increase performance. (`grid` gradient
  // is always computed.)
  auto input_requires_grad = output_mask[0];

  int64_t count = N * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      input.scalar_type(), "grid_sampler_2d_backward_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(grad_output)) {
        grid_sampler_2d_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<const scalar_t, int>(grad_output),
            getTensorInfo<const scalar_t, int>(input),
            getTensorInfo<const scalar_t, int>(grid),
            input_requires_grad ? getTensorInfo<scalar_t, int>(grad_input) : TensorInfo<scalar_t, int>(),
            getTensorInfo<scalar_t, int>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            /*grad_input_memory_span =*/input_requires_grad ? static_cast<int>(grad_input.numel()) : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        grid_sampler_2d_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<const scalar_t, int64_t>(grad_output),
            getTensorInfo<const scalar_t, int64_t>(input),
            getTensorInfo<const scalar_t, int64_t>(grid),
            input_requires_grad ? getTensorInfo<scalar_t, int64_t>(grad_input) : TensorInfo<scalar_t, int64_t>(),
            getTensorInfo<scalar_t, int64_t>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            /*grad_input_memory_span =*/input_requires_grad ? grad_input.numel() : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}


// std::tuple<Tensor, Tensor>
// grid_sampler_2d_backward_cuda(const Tensor& grad_output, const Tensor& input,
//                               const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
//                               bool align_corners, std::array<bool, 2> output_mask) {
//   auto input_requires_grad = output_mask[0];
//   Tensor grad_input = ([&]() {
//     if (input_requires_grad) {
//       return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
//     } else {
//       return Tensor();
//     }
//   })();
//   auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
//   launch_grid_sampler_2d_backward_kernel(
//       grad_input, grad_grid, grad_output, input,
//       grid, interpolation_mode, padding_mode, align_corners, output_mask);
//   return std::make_tuple(grad_input, grad_grid);
// }

// // Wrapper functions for export - forward uses PyTorch's grid_sampler_2d, backward uses our CUDA kernel
// Tensor grid_sample_forward(const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
//   return at::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
// }

// std::tuple<Tensor, Tensor> grid_sample_backward(const Tensor& grad_output, const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, std::array<bool, 2> output_mask) {
//   return grid_sampler_2d_backward_cuda(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
// }

Tensor grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());
  launch_grid_sampler_2d_forward_kernel(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

// Tensor grid_sampler_3d_cuda(const Tensor& input, const Tensor& grid,
//                             int64_t interpolation_mode, int64_t padding_mode,
//                             bool align_corners) {
//   auto in_size = input.sizes();
//   auto grid_size = grid.sizes();
//   auto output = at::empty(
//       {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
//       input.options());
//   launch_grid_sampler_3d_forward_kernel(
//       output, input, grid, interpolation_mode, padding_mode, align_corners);
//   return output;
// }

std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners, std::array<bool, 2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_grid_sampler_2d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

// Wrapper functions for export - export.cpp/export_hip.cpp bind these names
Tensor grid_sample_forward(const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  return grid_sampler_2d_cuda(input, grid, interpolation_mode, padding_mode, align_corners);
}

std::tuple<Tensor, Tensor> grid_sample_backward(const Tensor& grad_output, const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, std::array<bool, 2> output_mask) {
  return grid_sampler_2d_backward_cuda(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
}

// std::tuple<Tensor, Tensor>
// grid_sampler_3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
//                               const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
//                               bool align_corners, std::array<bool,2> output_mask) {
//   auto input_requires_grad = output_mask[0];
//   Tensor grad_input = ([&]() {
//     if (input_requires_grad) {
//       return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
//     } else {
//       return Tensor();
//     }
//   })();
//   auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
//   launch_grid_sampler_3d_backward_kernel(
//       grad_input, grad_grid, grad_output, input,
//       grid, interpolation_mode, padding_mode, align_corners, output_mask);
//   return std::make_tuple(grad_input, grad_grid);
// }

class GridSampleFunction : public torch::autograd::Function<GridSampleFunction> {
  public:
   static torch::autograd::variable_list forward(
     torch::autograd::AutogradContext* ctx,
     const torch::autograd::Variable& input,
     const torch::autograd::Variable& grid,
     int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
     auto pinput = input.expect_contiguous();
     auto pgrid = grid.expect_contiguous();
     auto ret = grid_sampler_2d_cuda(*pinput, *pgrid, interpolation_mode, padding_mode, align_corners);
     ctx->save_for_backward({*pinput, *pgrid});
     ctx->saved_data["interpolation_mode"] = interpolation_mode;
     ctx->saved_data["padding_mode"] = padding_mode;
     ctx->saved_data["align_corners"] = align_corners;
     return {ret};
   }
 
   static torch::autograd::variable_list backward(
       torch::autograd::AutogradContext* ctx,
       const torch::autograd::variable_list& gradOutput) {
     auto saved = ctx->get_saved_variables();
     auto input = saved[0];
     auto grid = saved[1];
     int64_t interpolation_mode = ctx->saved_data["interpolation_mode"].toInt();
     int64_t padding_mode = ctx->saved_data["padding_mode"].toInt();
     bool align_corners = ctx->saved_data["align_corners"].toBool();
     // Both input and grid require grad when backward is invoked (from autograd)
     std::array<bool, 2> output_mask = {true, true};
     auto ret = grid_sampler_2d_backward_cuda(gradOutput[0], input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
     return {std::get<0>(ret), std::get<1>(ret), torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable()};
   }
 };
 Tensor grid_sample_forward_autograd(const Tensor& input,const Tensor& grid,int64_t interpolation_mode,int64_t padding_mode,bool align_corners) {
    auto result = GridSampleFunction::apply(input, grid, interpolation_mode, padding_mode, align_corners);
    return result[0];
   }
}
}