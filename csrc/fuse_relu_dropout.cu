// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <curand_kernel.h>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>


namespace at {
    namespace native {

        namespace {

            /**
             * forward functions
             * */
// philox generates 128 bits of randomness at a time. Kernel uses this explicitly by putting suitably transformed result into float4
// for all members of float4 to be consumed UNROLL has to be 4. Don't change!
// Note: VEC <= 4 (and in most real-world cases will be 4), so same logic applies.
            const int UNROLL = 4;
            const int CUDABlockSize = 512;

            template<typename scalar_t, typename accscalar_t, typename IndexType,
                    int ADims, int VEC>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
            C10_LAUNCH_BOUNDS_2(CUDABlockSize, 4)
#endif
            __global__ void fused_dropout_kernel_vec(
                    at::cuda::detail::TensorInfo<scalar_t, IndexType> self,
                    at::cuda::detail::TensorInfo<scalar_t, IndexType> ret,
                    at::cuda::detail::TensorInfo<scalar_t, IndexType> retmask,
                    IndexType totalElements,
                    accscalar_t p,
                    bool use_relu,
                    bool training,
                    PhiloxCudaState philox_args) {

                // make sure we don't break assumption that we can't have > 4 elements / thread
                static_assert(VEC <= 4, "Value of VEC must be in [2, 4]");

                using LoadT = memory::aligned_vector<scalar_t, VEC>;
                using MaskLoadT = memory::aligned_vector<scalar_t, VEC>;

                IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
                bool is_negative = false;

                if (training) {
                    // set rand generator
                    auto seeds = at::cuda::philox::unpack(philox_args);
                    curandStatePhilox4_32_10_t state;
                    curand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

                    accscalar_t pinv = accscalar_t(1) / p;
                    // Helps align the total number of times curand_uniform4 is called by each thread for the same totalElements
                    // in the vec=2 and vec=4 cases.
                    bool gridxvec_loop_state = 0;

                    float4 rand;
                    // Note: Vectorized loads means we'll stride each thread by an additional VEC factor, as we'll load VEC elements at a time
                    for (IndexType linearIndex = idx * VEC; linearIndex < totalElements; linearIndex += gridDim.x * blockDim.x * VEC) {
                        // local storage
                        scalar_t src[VEC];
                        // We'll use this to actually cause vectorized loads later
                        LoadT *value = reinterpret_cast<LoadT *>(&src);

                        //curand_uniform_double was pure evil anyway, not doing what it promises, and there's nothing for halfs, so generate float for everything
                        // Note: need a new set of random values per 4 elements -- we'll handle VEC elements in this thread, so need ceil(VEC / 4)
                        // sets of rand.
                        if ((VEC == 4) || (gridxvec_loop_state == 0)) {
                            rand = curand_uniform4(&state);
                        } else {
                            // sets up the last two values we generated last iteration to be used this iteration.
                            rand.x = rand.z;
                            rand.y = rand.w;
                            gridxvec_loop_state ^= 1;
                        }
                        rand.x = rand.x < p;
                        rand.y = rand.y < p;
                        if (VEC == 4) {
                            rand.z = rand.z < p;
                            rand.w = rand.w < p;
                        }

                        // Note: We explicitly check for is_contiguous() before launching the vectorized kernel
                        // and replace IndexToOffset call with linearIndex to allow vectorization of NHWC (or other)
                        // ordering.
                        // Single vectorized load
                        *value = *reinterpret_cast<LoadT *>(&self.data[linearIndex]);

                        scalar_t r[VEC];
                        scalar_t mask[VEC];
                        // Perform the actual computation
#pragma unroll
                        for (int ii = 0; ii < VEC; ii++) {
                            if (use_relu) { is_negative = src[ii] <= scalar_t(0); }
                            r[ii] = (is_negative ? scalar_t(0) : src[ii]) * ((&rand.x)[ii]) * pinv;
                            mask[ii] = is_negative ? scalar_t(0) : scalar_t((&rand.x)[ii] * pinv);
                        }
                        // Vectorized writes for both mask & result
                        *(reinterpret_cast<LoadT *>(&ret.data[linearIndex])) = *reinterpret_cast<LoadT *>(&r[0]);
                        *(reinterpret_cast<MaskLoadT *>(&retmask.data[linearIndex])) = *reinterpret_cast<MaskLoadT *>(&mask[0]);

                        __syncthreads();
                    }
                } else {
                    // Note: Vectorized loads means we'll stride each thread by an additional VEC factor, as we'll load VEC elements at a time
                    for (IndexType linearIndex = idx * VEC; linearIndex < totalElements; linearIndex += gridDim.x * blockDim.x * VEC) {
                        // local storage
                        scalar_t src[VEC];
                        // We'll use this to actually cause vectorized loads later
                        LoadT *value = reinterpret_cast<LoadT *>(&src);
                        *value = *reinterpret_cast<LoadT *>(&self.data[linearIndex]);

                        scalar_t r[VEC];
                        scalar_t mask[VEC];
                        // Perform the actual computation
#pragma unroll
                        for (int ii = 0; ii < VEC; ii++) {
                            if (use_relu) { is_negative = src[ii] <= scalar_t(0); }
                            r[ii] = is_negative ? scalar_t(0) : src[ii];
                            mask[ii] = is_negative ? scalar_t(0) : scalar_t(1);
                        }
                        // Vectorized writes for both mask & result
                        *(reinterpret_cast<LoadT *>(&ret.data[linearIndex])) = *reinterpret_cast<LoadT *>(&r[0]);
                        *(reinterpret_cast<MaskLoadT *>(&retmask.data[linearIndex])) = *reinterpret_cast<MaskLoadT *>(&mask[0]);

                        __syncthreads();
                    }
                }
            }

            template<typename scalar_t, typename accscalar_t, typename IndexType, int ADims, int BDims = ADims>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
            C10_LAUNCH_BOUNDS_2(CUDABlockSize, 4)
#endif
            __global__ void fused_dropout_kernel(
                    cuda::detail::TensorInfo <scalar_t, IndexType> self,
                    cuda::detail::TensorInfo <scalar_t, IndexType> ret,
                    cuda::detail::TensorInfo <scalar_t, IndexType> retmask,
                    IndexType totalElements,
                    accscalar_t p,
                    bool use_relu,
                    bool training,
                    PhiloxCudaState philox_args) {

                IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
                bool is_negative = false;
                if (training) {
                    auto seeds = at::cuda::philox::unpack(philox_args);
                    curandStatePhilox4_32_10_t state;
                    curand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

                    accscalar_t pinv = accscalar_t(1) / p;
                    IndexType rounded_size = ((totalElements - 1) / (blockDim.x * gridDim.x * UNROLL) + 1) *
                                             blockDim.x * gridDim.x * UNROLL;

                    for (IndexType linearIndex = idx; linearIndex < rounded_size; linearIndex += gridDim.x * blockDim.x * UNROLL) {
//curand_uniform_double was pure evil anyway, not doing what it promises, and there's nothing for halfs, so generate float for everything
                        float4 rand = curand_uniform4(&state);
                        rand.x = rand.x < p;
                        rand.y = rand.y < p;
                        rand.z = rand.z < p;
                        rand.w = rand.w < p;
                        for (int ii = 0; ii < UNROLL; ii++) {
                            IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
                            if (li < totalElements) {
                                // Convert `linearIndex` into an offset of `ret`
                                const IndexType bOffset = cuda::detail::IndexToOffset<scalar_t, IndexType, BDims>::get(li, ret);
                                if (use_relu) { is_negative = self.data[bOffset] <= scalar_t(0); }
                                ret.data[bOffset] = (is_negative ? scalar_t(0) : self.data[bOffset]) * ((&rand.x)[ii]) * pinv;
                                retmask.data[bOffset] = is_negative ? scalar_t(0) : scalar_t((&rand.x)[ii] * pinv);
                            }
                        }
                        __syncthreads();
                    }

                } else {
                    IndexType rounded_size = ((totalElements - 1) / (blockDim.x * gridDim.x * UNROLL) + 1) *
                                             blockDim.x * gridDim.x * UNROLL;

                    for (IndexType linearIndex = idx; linearIndex < rounded_size; linearIndex += gridDim.x * blockDim.x * UNROLL) {
                        for (int ii = 0; ii < UNROLL; ii++) {
                            IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
                            if (li < totalElements) {
                                // Convert `linearIndex` into an offset of `ret`
                                const IndexType bOffset = cuda::detail::IndexToOffset<scalar_t, IndexType, BDims>::get(li, ret);
                                if (use_relu) { is_negative = self.data[bOffset] <= scalar_t(0); }
                                ret.data[bOffset] = (is_negative ? scalar_t(0) : self.data[bOffset]);
                                retmask.data[bOffset] = is_negative ? scalar_t(0) : scalar_t(1);
                            }
                        }
                        __syncthreads();
                    }
                }
            }

            /**
             * backward functions
             * */
            template<typename scalar_t, typename IndexType, int ADims, int BDims = ADims>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
            C10_LAUNCH_BOUNDS_2(CUDABlockSize, 4)
#endif
            __global__ void fused_dropout_kernel_backward(
                    at::cuda::detail::TensorInfo<scalar_t, IndexType> ret,
                    at::cuda::detail::TensorInfo<scalar_t, IndexType> grad,
                    at::cuda::detail::TensorInfo<scalar_t, IndexType> retmask,
                    IndexType totalElements) {

                IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
                IndexType rounded_size = ((totalElements - 1) / (blockDim.x * gridDim.x * UNROLL) + 1) * blockDim.x * gridDim.x * UNROLL;

                for (IndexType linearIndex = idx; linearIndex < rounded_size; linearIndex += gridDim.x * blockDim.x * UNROLL) {
                    for (int ii = 0; ii < UNROLL; ii++) {
                        IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
                        if (li < totalElements) {
                            // Convert `linearIndex` into an offset of `ret`
                            const IndexType bOffset = cuda::detail::IndexToOffset<scalar_t, IndexType, BDims>::get(li, ret);
                            ret.data[bOffset] = grad.data[bOffset] * retmask.data[bOffset];
                        }
                    }
                    __syncthreads();
                }
            }


            /**
             * forward fused_relu_dropout_forward_launcher
             */
            template<typename scalar_t>
            int get_vector_size(at::Tensor self, at::Tensor ret, at::Tensor mask) {
                int vec_size = 4;
                // get the vector size
                if (!self.is_non_overlapping_and_dense() || !ret.is_non_overlapping_and_dense() || !mask.is_non_overlapping_and_dense()) {
                    vec_size = 1;
                } else {
                    vec_size = memory::can_vectorize_up_to<scalar_t>((char *) self.data_ptr());
                }

                // check that we'd have no remainders - prefer a smaller vector size with no remainders over a larger vector and remainder.
                bool can_vectorize = true;
                do {
                    can_vectorize = self.numel() % vec_size == 0 && ret.numel() % vec_size == 0 && mask.numel() % vec_size == 0;
                    if (!can_vectorize) vec_size /= 2;
                } while (vec_size > 1 && !can_vectorize);
                return can_vectorize ? vec_size : 1;
            }

            template<typename index_type>
            inline void fused_relu_dropout_forward_launcher(
                    const Tensor &self,
                    Tensor &ret,
                    Tensor &mask,
                    double p,
                    bool use_relu,
                    const int64_t nelem,
                    const PhiloxCudaState rng_engine_inputs,
                    bool training,
                    dim3 grid,
                    dim3 dim_block) {
                AT_DISPATCH_FLOATING_TYPES_AND2(
                        at::ScalarType::Half,
                        at::ScalarType::BFloat16,
                        self.scalar_type(),
                        "fused_relu_dropout",
                        [&] {
                            using accscalar_t = acc_type<scalar_t, true>;
                            accscalar_t pa = (accscalar_t) (p);
                            auto self_info = cuda::detail::getTensorInfo<scalar_t, index_type>(self);
                            auto ret_info = cuda::detail::getTensorInfo<scalar_t, index_type>(ret);
                            auto mask_info = cuda::detail::getTensorInfo<scalar_t, index_type>(mask);
                            self_info.collapseDims();
                            ret_info.collapseDims();
                            mask_info.collapseDims(); // ret and mask are collapsed to 1d
                            // contiguous tensor

                            int vec_size = get_vector_size<scalar_t>(self, ret, mask);
                            if (vec_size > 1) {
                                switch (vec_size) {
                                    case 4:
                                        fused_dropout_kernel_vec<scalar_t, accscalar_t, index_type, 1, 4>
                                        <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                                self_info, ret_info, mask_info, nelem, pa, use_relu, training, rng_engine_inputs);
                                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                                        break;
                                    case 2:
                                        fused_dropout_kernel_vec<scalar_t, accscalar_t, index_type, 1, 2>
                                        <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                                self_info, ret_info, mask_info, nelem, pa, use_relu, training, rng_engine_inputs);
                                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                                        break;
                                }
                            } else {
                                switch (self_info.dims) {
                                    case 1:
                                        fused_dropout_kernel<scalar_t, accscalar_t, index_type, 1>
                                        <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                                self_info, ret_info, mask_info, nelem, pa, use_relu, training, rng_engine_inputs);
                                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                                        break;
                                    default:
                                        if (!self.is_contiguous() && ret.is_contiguous() && mask.is_contiguous()) {
                                            fused_dropout_kernel<scalar_t, accscalar_t, index_type, -1, 1>
                                            <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                                    self_info, ret_info, mask_info, nelem, pa, use_relu, training, rng_engine_inputs);
                                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                                        } else {
                                            fused_dropout_kernel<scalar_t, accscalar_t, index_type, -1>
                                            <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                                    self_info, ret_info, mask_info, nelem, pa, use_relu, training, rng_engine_inputs);
                                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                                        }
                                }
                            }
                        });
            }

            template<typename index_type>
            inline void fused_relu_dropout_backward_launcher(
                    Tensor &ret,
                    const Tensor &grad,
                    const Tensor &mask,
                    const int64_t &nelem,
                    dim3 grid,
                    dim3 dim_block) {
                AT_DISPATCH_FLOATING_TYPES_AND2(
                        at::ScalarType::Half,
                        at::ScalarType::BFloat16,
                        grad.scalar_type(),
                        "fused_relu_dropout_backward",
                        [&] {
                            auto ret_info = cuda::detail::getTensorInfo<scalar_t, index_type>(ret);
                            auto grad_info = cuda::detail::getTensorInfo<scalar_t, index_type>(grad);
                            auto mask_info = cuda::detail::getTensorInfo<scalar_t, index_type>(mask);
                            // collapsed to 1d
                            ret_info.collapseDims();
                            grad_info.collapseDims();
                            mask_info.collapseDims();
                            // contiguous tensor

                            fused_dropout_kernel_backward<scalar_t, index_type, 1>
                            <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                    ret_info, grad_info, mask_info, nelem);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
            }

        } //anonymous namespace


        std::tuple <Tensor, Tensor>
        fused_relu_dropout_forward(const Tensor &self, const double &p, const bool &use_relu, const bool &training) {
            auto gen = get_generator_or_default<CUDAGeneratorImpl>(Generator(), cuda::detail::getDefaultCUDAGenerator());
            Tensor ret = at::empty_like(self);
            Tensor mask = at::empty_like(self);
            const int64_t nelem = self.numel();
//empty tensors should not get here, but just in case, avoid FPE
            if (nelem == 0) return std::tuple<Tensor, Tensor>(self, mask);

            unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / CUDABlockSize;
            dim3 dim_block(CUDABlockSize);
            dim3 grid((nelem + CUDABlockSize - 1) / CUDABlockSize);
            grid.x = std::min((unsigned int) at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);

//number of times random will be generated per thread, to offset philox counter in thc random state
            int64_t counter_offset = ((nelem - 1) / (CUDABlockSize * grid.x * UNROLL) + 1) * UNROLL;
            PhiloxCudaState rng_engine_inputs;
            {
                // See Note [Acquire lock when using random generators]
                std::lock_guard <std::mutex> lock(gen->mutex_);
                rng_engine_inputs = gen->philox_cuda_state(counter_offset);
            }

            if (cuda::detail::canUse32BitIndexMath(self)) {
                fused_relu_dropout_forward_launcher<unsigned int>(
                        self, ret, mask, p, use_relu, nelem, rng_engine_inputs, training, grid, dim_block);
            } else {
                fused_relu_dropout_forward_launcher<uint64_t>(self, ret, mask, p, use_relu, nelem, rng_engine_inputs, training, grid, dim_block);
            }

            return std::tuple<Tensor, Tensor>(ret, mask);
        }

        Tensor fused_relu_dropout_backward(Tensor &grad, Tensor &mask) {
            Tensor ret = at::empty_like(grad, grad.suggest_memory_format());
            const int64_t nelem = grad.numel();
//empty tensors should not get here, but just in case, avoid FPE
            if (nelem == 0) return ret;

            unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / CUDABlockSize;
            dim3 dim_block(CUDABlockSize);
            dim3 grid((nelem + CUDABlockSize - 1) / CUDABlockSize);
            grid.x = std::min((unsigned int) at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);

            if (cuda::detail::canUse32BitIndexMath(grad)) {
                fused_relu_dropout_backward_launcher<unsigned int>(ret, grad, mask, nelem, grid, dim_block);
            } else {
                fused_relu_dropout_backward_launcher<uint64_t>(ret, grad, mask, nelem, grid, dim_block);
            }

            return ret;
        }

    } // namespace native
} // namespace at
