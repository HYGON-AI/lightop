#pragma once

#include <THC/THCDeviceUtils.cuh>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#include <ATen/native/cuda/LaunchUtils.h>
#include <c10/macros/Macros.h>

#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <curand_kernel.h>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <torch/extension.h>

#include <bn_common.h>
namespace at { namespace native {
const int UNROLL = 4;
#define VEC 4
typedef unsigned int IndexType;

template <typename scalar_t, typename accscalar_t, typename PTA, typename PTA1>
struct GradOp {
  __device__ GradOp(accscalar_t m, const PTA& i, const PTA& g, const PTA1& o, accscalar_t p)
    : mean(m), input(i), grad_output(g), output(o), pa(p){}
  __device__ __forceinline__ Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n] * output[batch][plane][n] / pa;
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
  const PTA1& output;
  const accscalar_t pa;
};

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, bool train, typename index_t>
__global__ void bn_dropout_transform_input_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> input,
    GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> output,
    GenericPackedTensorAccessor<bool, 3, RestrictPtrTraits, index_t> out_dropout,
    const GenericPackedTensorAccessor<typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::type, 1, RestrictPtrTraits, index_t> mean_,
    const GenericPackedTensorAccessor<typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::type, 1, RestrictPtrTraits, index_t> var_or_invstd,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> weight,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> bias,
    stat_accscalar_t epsilon,
    const stat_accscalar_t pa,
    const int64_t nelem,
    const PhiloxCudaState philox_args,
    const IndexType hw,
    const IndexType chw) {
    
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nelem) {
    return;
  }
  curandStatePhilox4_32_10_t state;
  if (pa != 1) {
    auto seeds = at::cuda::philox::unpack(philox_args);
    curand_init(std::get<0>(seeds),
                idx,
                std::get<1>(seeds),
                &state);
  }

  stat_accscalar_t pinv = stat_accscalar_t(1)/pa;

  float4 rand;
  for (IndexType linearIndex = idx * VEC;
      linearIndex < nelem;
      linearIndex += gridDim.x * blockDim.x * VEC) {
    if (pa != 1) {
      rand = curand_uniform4(&state);
      rand.x = rand.x < pa;
      rand.y = rand.y < pa;
      rand.z = rand.z < pa;
      rand.w = rand.w < pa;
    } else {
      rand.x = 1;
      rand.y = 1;
      rand.z = 1;
      rand.w = 1;
    }

    for (int ii = 0; ii < VEC; ii++) {
      IndexType index = linearIndex + ii;
      if(index >= nelem) {
        continue;
      }
      IndexType batch = index / chw;
      IndexType plane = (index % chw) / hw;
      IndexType hw_index = index - (chw * batch + hw * plane);

      stat_accscalar_t gamma = weight.size(0) > 0 ? static_cast<stat_accscalar_t>(weight[plane]) : static_cast<stat_accscalar_t>(1);
      stat_accscalar_t beta = bias.size(0) > 0 ? static_cast<stat_accscalar_t>(bias[plane]) : static_cast<stat_accscalar_t>(0);
      stat_accscalar_t mean = static_cast<stat_accscalar_t>(mean_[plane]);
      stat_accscalar_t invstd;
      if (train) {
        invstd = var_or_invstd[plane];
      } else {
        invstd = static_cast<stat_accscalar_t>(1) / device_sqrt(static_cast<stat_accscalar_t>(var_or_invstd[plane]) + epsilon);
      }
      // printf("batch:%d plane:%d hw_index:%d data:%f\n", batch, plane, hw_index, static_cast<stat_accscalar_t>(input[batch][plane][hw_index]));
      auto tmp = static_cast<input_scalar_t>(gamma * (input[batch][plane][hw_index] - mean) * invstd + beta);
      output[batch][plane][hw_index] = static_cast<input_scalar_t>((&rand.x)[ii] * tmp * pinv);
      out_dropout[batch][plane][hw_index] = (&rand.x)[ii];
    }
  }
}
template <typename VarTransform, typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void bn_dropout_collect_statistics_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> input,
    const stat_accscalar_t epsilon,
    const stat_accscalar_t momentum,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_mean,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_transformed_var) {

  __shared__ int shared_n[2 * 2 * C10_WARP_SIZE + C10_WARP_SIZE];

  int plane = blockIdx.x;
  int N = input.size(0) * input.size(2);
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Compute the mean and variance across (batch, x/y/z)
  // this uses the Welford (in the for loop)/parallel algorithm (to sum across the block)
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
  // and the parallel algorithm on the same page.
  // We use two shuffles to reduce across the entire block.
  // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a description.
  stat_accscalar_t* shared_avg_var = (stat_accscalar_t*) &shared_n[C10_WARP_SIZE];

  // first the reductions each thread does separately
  stat_accscalar_t avg = 0;
  stat_accscalar_t var_n = 0;
  int n = 0;
  for (int batch = threadIdx.y; batch < input.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < input.size(2); x += blockDim.x) {
      stat_accscalar_t v = input[batch][plane][x];
      stat_accscalar_t d1 = v - avg;
      n++;
      avg += d1 / n;
      var_n += d1 * (v - avg);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  for (int i = 0; i < getMSB(C10_WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = __shfl_xor(avg, 1 << i, C10_WARP_SIZE);
    int o_n = __shfl_xor(n, 1 << i, C10_WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += __shfl_xor(var_n, 1 << i, C10_WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // this writes each warps  item into shared memory
  // there are at most C10_WARP_SIZE items left because
  // there are at most C10_WARP_SIZE**2 threads at the beginning
  __syncthreads();
  if (tid % C10_WARP_SIZE == 0) {
    shared_n[tid / C10_WARP_SIZE] = n;
    shared_avg_var[tid / C10_WARP_SIZE * 2] = avg;
    shared_avg_var[tid / C10_WARP_SIZE * 2 + 1] = var_n;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid < C10_WARP_SIZE) {
    n = (tid < blockDim.x * blockDim.y / C10_WARP_SIZE ? shared_n[tid] : 0);
    avg = (tid < blockDim.x * blockDim.y  / C10_WARP_SIZE ? shared_avg_var[2 * tid] : stat_accscalar_t(0));
    var_n = (tid < blockDim.x * blockDim.y  / C10_WARP_SIZE ? shared_avg_var[2 * tid + 1] : stat_accscalar_t(0));
  }
  for (int i = 0; i < getMSB(C10_WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = __shfl_xor(avg, 1 << i, C10_WARP_SIZE);
    int o_n = __shfl_xor(n, 1 << i, C10_WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += __shfl_xor(var_n, 1 << i, C10_WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // Save the mean, variance, and moving averages
  if (tid == 0) {
    if (save_mean.data() != NULL) {
      save_mean[plane] = avg;
    }
    if (save_transformed_var.data() != NULL) {
      save_transformed_var[plane] = VarTransform{}(var_n / N, epsilon);
    }
  }

}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void bn_dropout_backward_res(
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> input,
    const GenericPackedTensorAccessor<bool, 3, DefaultPtrTraits, index_t> output,
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> save_mean,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> running_mean,
    bool train,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_res_v1,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_res_v2,
    stat_accscalar_t pa) {

  index_t plane = blockIdx.x;
  stat_accscalar_t mean;
  if (train) {
    mean = save_mean[plane];
  } else {
    mean = static_cast<stat_accscalar_t>(running_mean[plane]);
  }
  GradOp<input_scalar_t, stat_accscalar_t, GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>,GenericPackedTensorAccessor<bool, 3, DefaultPtrTraits, index_t>> g(mean, input, grad_output, output, pa);
  auto res = reduce<Float2<input_scalar_t, stat_accscalar_t>>(g, grad_output, plane);

  grad_res_v1[plane] = static_cast<stat_accscalar_t>(res.v1);
  grad_res_v2[plane] = static_cast<stat_accscalar_t>(res.v2);
}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void bn_dropout_backward_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> input,
    const GenericPackedTensorAccessor<bool, 3, DefaultPtrTraits, index_t> output,
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
    stat_accscalar_t pa) {

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
    //for (int batch = threadIdx.y; batch < grad_output.size(0); batch += blockDim.y) {
      index_t batch = blockIdx.y;
      for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
        bool o = output[batch][plane][x];
        input_scalar_t go = grad_output[batch][plane][x] * o / pa;
        if (train) {
          stat_accscalar_t inp = input[batch][plane][x];
          stat_accscalar_t proj = (inp - mean) * proj_scale;
          grad_input[batch][plane][x] = static_cast<input_scalar_t>((go - proj - grad_mean) * grad_scale);
        } else {
          grad_input[batch][plane][x] = static_cast<input_scalar_t>(go * grad_scale);
        }
      }
    //}
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

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor> bn_dropout_backward_cuda_template(const Tensor& grad_out_, const Tensor& input_, const Tensor& output_, 
          const Tensor& weight_, const Tensor& running_mean_, const Tensor& running_var_, const Tensor& save_mean_, const Tensor& save_invstd_,
          bool train, double epsilon, std::array<bool,3> grad_input_mask, double p) {

  using accscalar_t = at::acc_type<stat_scalar_t, true>;
  Tensor grad_input_;
  Tensor grad_input_reshaped;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1});
  auto output_reshaped = output_.reshape({output_.size(0), output_.size(1), -1});
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  if (grad_input_mask[0]) {
    grad_input_ = at::empty_like(input_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    grad_input_reshaped = grad_input_.view(input_reshaped.sizes());
  }
  if (grad_input_mask[1]) {
    grad_weight_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    grad_bias_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  auto input = get_packed_accessor<
      input_scalar_t, 3, DefaultPtrTraits, index_t>(input_reshaped, "input");
  auto grad_output = get_packed_accessor<
      input_scalar_t, 3, DefaultPtrTraits, index_t>(grad_output_reshaped, "grad_output");
  auto grad_input = packed_accessor_or_dummy<
      input_scalar_t, 3, DefaultPtrTraits, index_t>(grad_input_reshaped, "grad_input");
  auto weight = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(weight_, "weight");
  auto grad_weight = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(grad_weight_, "grad_weight");
  auto grad_bias = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(grad_bias_, "grad_bias");
  auto running_mean = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(running_mean_, "running_mean");
  auto running_var = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(running_var_, "running_var");
  auto save_mean = packed_accessor_or_dummy<
      accscalar_t, 1, DefaultPtrTraits, index_t>(save_mean_, "save_mean");
  auto save_invstd = packed_accessor_or_dummy<
      accscalar_t, 1, DefaultPtrTraits, index_t>(save_invstd_, "save_invstd");

  auto stream1 = at::cuda::getCurrentCUDAStream();
  dim3 blocks_res(input.size(1));
  // int tf1 = getNumThreads_1024(input.size(2));
  // dim3 threads1(tf1, std::max<int>(1, MAX_BLOCK_SIZE_1024/tf1));
  int tf1 = getNumThreads(input.size(2));
  dim3 threads1(tf1, std::max<int>(1, MAX_BLOCK_SIZE/tf1));

  auto options = input_.options().dtype(
      at::toAccumulateType(input_.scalar_type(), /*is_cuda=*/true));
  int64_t n_channels = input_.size(1);
  Tensor grad_res_v1_;
  Tensor grad_res_v2_;
  if (grad_input_mask[1]) {
      grad_res_v1_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      grad_res_v2_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  } else {
      grad_res_v1_ = at::empty_like(running_mean_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      grad_res_v2_ = at::empty_like(running_mean_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  auto grad_res_v1 = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(grad_res_v1_, "grad_res_v1");
  auto grad_res_v2 = packed_accessor_or_dummy<
      stat_scalar_t, 1, DefaultPtrTraits, index_t>(grad_res_v2_, "grad_res_v2");

  auto stream = at::cuda::getCurrentCUDAStream();
  //dim3 blocks(input.size(1));
  dim3 blocks(input.size(1), grad_output.size(0));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf);
  accscalar_t pa = 1 - p;
  auto output = get_packed_accessor<
    bool, 3, DefaultPtrTraits, index_t>(output_reshaped, "output");
  bn_dropout_backward_res<input_scalar_t,stat_scalar_t, accscalar_t, index_t> <<<blocks_res, threads1, 0, stream1>>>
    (input,output,grad_output,save_mean,running_mean,train,grad_res_v1, grad_res_v2, pa);

  bn_dropout_backward_kernel<input_scalar_t, stat_scalar_t, accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
    (input, output, grad_output, grad_input, grad_weight, grad_bias, weight, running_mean, running_var,
     save_mean, save_invstd, train, epsilon, grad_res_v1, grad_res_v2, pa);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(grad_input_, grad_weight_, grad_bias_);
}

template<typename scalar_t, typename index_t, typename VarTransform>
void bn_dropout_stats_cuda_template(
    const Tensor& out_mean, const Tensor& out_invstd, const Tensor& input_, double epsilon) {

  using accscalar_t = at::acc_type<scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions

  resize_output(out_mean, {n_input});
  resize_output(out_invstd, {n_input});
  auto input = get_packed_accessor<
      scalar_t, 3, RestrictPtrTraits, index_t>(input_reshaped, "input");
  TORCH_INTERNAL_ASSERT(out_invstd.dim() == 1 && out_invstd.is_contiguous() &&
                        out_invstd.sizes()[0]);
  TORCH_INTERNAL_ASSERT(out_mean.dim() == 1 && out_mean.is_contiguous() &&
                        out_mean.sizes()[0]);

  auto mean = packed_accessor_or_dummy<
      accscalar_t, 1, RestrictPtrTraits, index_t>(out_mean, "out_mean");
  auto invstd = packed_accessor_or_dummy<
      accscalar_t, 1, RestrictPtrTraits, index_t>(out_invstd, "out_invstd");
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(input.size(1));
  // int tf = getNumThreads_1024(input.size(2));
  // dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE_1024/tf));

  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

  bn_dropout_collect_statistics_kernel<VarTransform, scalar_t, scalar_t, accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
    (input, epsilon, 0.0, mean, invstd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
void bn_dropout_elemt_cuda_template(const Tensor& output_, const Tensor& out_dropout_, const Tensor& input_, const Tensor& weight_,
                                    const Tensor& bias_, const Tensor& mean_, const Tensor& invstd_, double p) {

  using stat_accscalar_t = at::acc_type<stat_scalar_t, true>;
  int64_t n_input = input_.size(1);
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions
  auto output_reshaped = output_.view({input_.size(0), input_.size(1), -1});

  auto input = get_packed_accessor<
      input_scalar_t, 3, RestrictPtrTraits, index_t>(input_reshaped, "input");
  auto output = get_packed_accessor<
      input_scalar_t, 3, RestrictPtrTraits, index_t>(output_reshaped, "output");
  auto weight = packed_accessor_or_dummy<
    stat_scalar_t, 1, RestrictPtrTraits, index_t>(weight_, "weight");
  auto bias = packed_accessor_or_dummy<
      stat_scalar_t, 1, RestrictPtrTraits, index_t>(bias_, "bias");
  auto mean = packed_accessor_or_dummy<
      stat_accscalar_t, 1, RestrictPtrTraits, index_t>(mean_, "mean");
  auto invstd = packed_accessor_or_dummy<
      stat_accscalar_t, 1, RestrictPtrTraits, index_t>(invstd_, "invstd");
  auto stream = at::cuda::getCurrentCUDAStream();

  // NOTE: We use transform_input_kernel in training mode, which ignores epsilon
  const double dummy_epsilon = 1e-5;

  const int64_t nelem = input_.numel();   
  IndexType hw = input.size(2);
  IndexType chw = input.size(1) * hw;
  const int64_t block_size = 256;
  dim3 dim_block(block_size);
  dim3 grid((nelem + block_size -1)/block_size);
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(Generator(), cuda::detail::getDefaultCUDAGenerator());
  IndexType blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
  grid.x = std::min((IndexType)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);
  int64_t counter_offset = ((nelem - 1)/(block_size*grid.x*UNROLL)+1)*UNROLL;
  auto out_dropout_reshaped = out_dropout_.view({input_.size(0), input_.size(1), -1});
  auto out_dropout = get_packed_accessor<
      bool, 3, RestrictPtrTraits, index_t>(out_dropout_reshaped, "out_dropout");
  PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }
  stat_accscalar_t pa = (stat_accscalar_t)p;
  
  bn_dropout_transform_input_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, true, index_t> <<<grid, dim_block, 0, stream>>>
    (input, output, out_dropout, mean, invstd, weight, bias, dummy_epsilon, pa, nelem, rng_engine_inputs, hw, chw);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} } // namespace at::native
