#pragma once

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#include <ATen/native/cuda/LaunchUtils.h>
#include <c10/macros/Macros.h>

namespace at { namespace native {

enum fuse_action_mode {
  fuse_act_relu = 1,
  fuse_act_silu = 2
};
template <typename input_scalar_t>
__device__ __forceinline__ input_scalar_t act_forward(input_scalar_t o_bn, int act_mode)
{
  return (o_bn / (static_cast<input_scalar_t>(1.0) + c10::cuda::compat::exp(-o_bn)));
}

template <typename input_scalar_t>
__device__ __forceinline__ input_scalar_t act_backward(input_scalar_t o_bn, input_scalar_t o_grad, int act_mode)
{
  const input_scalar_t dy_acc = static_cast<input_scalar_t>(o_grad);
  const input_scalar_t x_acc = static_cast<input_scalar_t>(o_bn);
  const input_scalar_t s_acc = input_scalar_t(1) / (input_scalar_t(1) + c10::cuda::compat::exp(-x_acc));
  return (dy_acc * s_acc * (input_scalar_t(1) + x_acc * (input_scalar_t(1) - s_acc)));
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB_act(int val) {
  return 31 - __clz(val);
}

template <typename scalar_t, typename accscalar_t>
struct Float2_act {
  accscalar_t v1, v2;
  __device__ Float2_act() {}
  __device__ Float2_act(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  __device__ Float2_act(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  __device__ Float2_act& operator+=(const Float2_act& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp_act {
  __device__ GradOp_act(accscalar_t m, const PTA& i, const PTA& g, const PTA& o, int mode)
    : mean(m), input(i), grad_output(g), output(o), act_mode(mode) {}
  __device__ __forceinline__ Float2_act<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = static_cast<accscalar_t>(act_backward<scalar_t>(output[batch][plane][n], grad_output[batch][plane][n], act_mode));
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2_act<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const int act_mode;
  const PTA& input;
  const PTA& grad_output;
  const PTA& output;
};

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum_act(T val) {
  for (int i = 0; i < getMSB_act(C10_WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, C10_WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, typename accscalar_t>
static __device__ __forceinline__ Float2_act<scalar_t, accscalar_t> warpSum_act(Float2_act<scalar_t, accscalar_t> value) {
  value.v1 = warpSum_act(value.v1);
  value.v2 = warpSum_act(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
// this works by first having each thread sum it's part
// of the data. Then there is a double-shuffling reduction.
// First each warp (of C10_WARP_SIZE threads) uses warpSum to reduce its
// data to the "warp leader", who writes its value into shared memory.
// Then a single warp reads the remaining (at most C10_WARP_SIZE) items
// and reduces them using another warpSum.
// The implicit assumption is that there are no more
// than C10_WARP_SIZE**2 threads.
template<typename scalar_t, typename Op, typename PTA>
__device__ scalar_t reduce_act(Op op, PTA tensor, int plane) {
  // first the reductions each thread does separately
  scalar_t sum = static_cast<scalar_t>(0);
  for (int batch = threadIdx.y; batch < tensor.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  sum = warpSum_act(sum);

  // this writes each warps  item into shared memory
  // there are at most C10_WARP_SIZE items left because
  // there are at most C10_WARP_SIZE**2 threads at the beginning
  __shared__ scalar_t shared[C10_WARP_SIZE];
  __syncthreads();
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (tid % C10_WARP_SIZE == 0) {
    shared[tid / C10_WARP_SIZE] = sum;
  }
  if (tid >= blockDim.x * blockDim.y / C10_WARP_SIZE && tid < C10_WARP_SIZE) {
    // zero out the other entries in shared
    shared[tid] = (scalar_t)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid / C10_WARP_SIZE == 0) {
    sum = warpSum_act(shared[tid]);
    if (tid == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}


template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, bool train, typename index_t>
__global__ void bnact_transform_input_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> input,
    GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> output,
    GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> output_bn,
    const GenericPackedTensorAccessor<typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::type, 1, RestrictPtrTraits, index_t> mean_,
    const GenericPackedTensorAccessor<typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::type, 1, RestrictPtrTraits, index_t> var_or_invstd,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> weight,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> bias,
    stat_accscalar_t epsilon,
    const int fuse_act_mode) {
  index_t plane = blockIdx.x;

  if (plane >= input.size(1)) {
    return;
  }

  stat_accscalar_t gamma = weight.size(0) > 0 ? static_cast<stat_accscalar_t>(weight[plane]) : static_cast<stat_accscalar_t>(1);
  stat_accscalar_t beta = bias.size(0) > 0 ? static_cast<stat_accscalar_t>(bias[plane]) : static_cast<stat_accscalar_t>(0);
  stat_accscalar_t mean = static_cast<stat_accscalar_t>(mean_[plane]);
  stat_accscalar_t invstd;
  if (train) {
    invstd = var_or_invstd[plane];
  } else {
    invstd = static_cast<stat_accscalar_t>(1) / device_sqrt(static_cast<stat_accscalar_t>(var_or_invstd[plane]) + epsilon);
  }

  index_t bs = input.size(0);
  index_t fs = input.size(2);

  index_t bstep  = blockDim.y * gridDim.y;
  for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep) {
    auto o = output[batch][plane];
    auto i = input[batch][plane];
    auto bn_o = output_bn[batch][plane];
    for (index_t feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      input_scalar_t tmp = static_cast<input_scalar_t>(gamma * (i[feature] - mean) * invstd + beta);
      bn_o[feature] = tmp;
      o[feature] = act_forward<input_scalar_t>(tmp, fuse_act_mode);
    }
  }
}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_act_backward_res(
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> input,
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> output,
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> save_mean,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> running_mean,
    bool train,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_res_v1,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_res_v2,
    const int act_mode) {

  index_t plane = blockIdx.x;
  stat_accscalar_t mean;
  if (train) {
    mean = save_mean[plane];
  } else {
    mean = static_cast<stat_accscalar_t>(running_mean[plane]);
  }
  GradOp_act<input_scalar_t, stat_accscalar_t, GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>> g_act(mean, input, grad_output, output, act_mode);
  auto res = reduce_act<Float2_act<input_scalar_t, stat_accscalar_t>>(g_act, grad_output, plane);

  grad_res_v1[plane] = static_cast<stat_accscalar_t>(res.v1);
  grad_res_v2[plane] = static_cast<stat_accscalar_t>(res.v2);
}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void bnact_backward_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> input,
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> output,
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_input,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_weight,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_bias,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> weight,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> running_mean,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> running_var,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> save_mean,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> save_invstd,
    bool train,
    stat_accscalar_t epsilon,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_res_v1,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_res_v2,
    const int act_mode) {

  index_t plane = blockIdx.x;
  index_t N = grad_output.size(0) * grad_output.size(2);

  stat_accscalar_t mean, invstd;
  if (train) {
    mean = save_mean[plane];
    invstd = save_invstd[plane];
  } else {
    mean = static_cast<stat_accscalar_t>(running_mean[plane]);
    invstd = static_cast<stat_accscalar_t>(1) / device_sqrt(static_cast<stat_accscalar_t>(running_var[plane]) + epsilon);
  }

  stat_accscalar_t weight_val = weight.size(0) > 0 ? static_cast<stat_accscalar_t>(weight[plane]) : stat_accscalar_t(1);
  stat_accscalar_t norm = stat_accscalar_t(1) / N;

  stat_accscalar_t grad_output_sum = static_cast<stat_accscalar_t>(grad_res_v1[plane]);
  stat_accscalar_t dot_p = static_cast<stat_accscalar_t>(grad_res_v2[plane]);

  stat_accscalar_t grad_mean = grad_output_sum * norm;
  stat_accscalar_t proj_scale = dot_p * norm * invstd * invstd;
  stat_accscalar_t grad_scale = invstd * weight_val;

  if (grad_input.data() != NULL) {
      index_t batch = blockIdx.y;
      for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
        input_scalar_t go = act_backward<input_scalar_t>(output[batch][plane][x], grad_output[batch][plane][x], act_mode);
        
        if (train) {
          stat_accscalar_t inp = input[batch][plane][x];
          stat_accscalar_t proj = (inp - mean) * proj_scale;
          grad_input[batch][plane][x] = static_cast<input_scalar_t>((go - proj - grad_mean) * grad_scale);
        } else {
          grad_input[batch][plane][x] = static_cast<input_scalar_t>(go * grad_scale);
        }
      }
  }

  if (grad_weight.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_weight[plane] = static_cast<stat_scalar_t>(dot_p * invstd);
    }
  }

  if (grad_bias.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_bias[plane] = static_cast<stat_scalar_t>(grad_output_sum);
    }
  }
}

} } // namespace at::native
