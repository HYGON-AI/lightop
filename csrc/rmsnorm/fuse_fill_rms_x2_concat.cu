// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/AccumulateType.h>
#include <THC/THCDeviceUtils.cuh>
#include <ATen/Dispatch.h>
#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif


#include "lightop_warp_size.h"

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
template <typename T, int N> struct alignas(sizeof(T) * N) Vector {
  T val[N];
};







namespace at{
    namespace native{

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

        
    template <typename T,int reducesize=C10_WARP_SIZE>
    __inline__ __device__ T WarpReduceSum_NEW(T val) {
    #pragma unroll
      for (int offset = reducesize/2; offset > 0; offset >>= 1) {
        val += WARP_SHFL_DOWN(val, offset);
      }
      return val;
    }
    
    template <typename T,int block_size=512>
    __inline__ __device__ T BlockReduceSum_NEW(T val, T* shared) {
      constexpr int share_size=block_size/C10_WARP_SIZE;
      val = WarpReduceSum_NEW<T>(val);
      if constexpr(block_size==C10_WARP_SIZE)
      {
        return val;
      }
      else{
        const int lid = threadIdx.x % C10_WARP_SIZE;
        const int wid = threadIdx.x / C10_WARP_SIZE;
        if (lid == 0&&wid<share_size) {
          shared[wid] = val;
        }
        __syncthreads();
        if (wid == 0&&lid<share_size) {
          val = WarpReduceSum_NEW<T,share_size>(shared[lid]);
        }
        return val;
      }
    }


    template <typename scalar_t,typename T_ACC,int Vec=4,int block_size=512>
    __global__ void fused_fill_rms_x2_concat_kernel(
        int64_t* positions,
        scalar_t* input,
        scalar_t* output,
        scalar_t* gamma,
        int cols,
        T_ACC eps,
        int stride,
        int num_tokens_inputs_embeds,
        scalar_t* previous_data,
        scalar_t* previous_out_data,
        scalar_t* previous_weight_data)

    {
      if(blockIdx.x<num_tokens_inputs_embeds)
      {
        int i=blockIdx.x;
        bool is_first_token = positions[i] == 0;
        if(is_first_token)
        {
            using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
            scalar_t intput_vec[Vec];
            #pragma unroll
            for(int ii=0;ii<Vec;ii++){
            intput_vec[ii] = static_cast<scalar_t>(0);
            }
            int j=threadIdx.x;
            int64_t col_offset = j * Vec;
            int out_cols = cols * 2;
            int64_t out_idx = i * out_cols + col_offset;
            int tcol=cols/Vec;
            if (j < tcol)
            {
                *(LoadT*)(output+out_idx)=*(LoadT*)intput_vec;
            }
            return;
        }   
        else  
        { 
            constexpr int share_size=block_size/C10_WARP_SIZE;
            __shared__ T_ACC val_shared[share_size];
            __shared__ T_ACC s_rstd;
            T_ACC val=0;
            int j=threadIdx.x;
            int tcol=cols/Vec;
            int out_cols = cols * 2;
            using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
            scalar_t intput_vec[Vec];
            T_ACC trstd;
            int64_t row_start_offset = i * stride;
            int64_t col_offset = j * Vec;
            int64_t idx = row_start_offset + col_offset;
            int64_t out_idx = i * out_cols + col_offset;
            
            if (j < tcol) 
            {
                *(LoadT*)intput_vec = *(LoadT*)(input+idx);
                #pragma unroll
                for (int ii = 0; ii < Vec; ii++) {
                val += static_cast<T_ACC>(intput_vec[ii])*static_cast<T_ACC>(intput_vec[ii]);
                }
            }
            val = BlockReduceSum_NEW<T_ACC,block_size>(val,val_shared);
            if (j == 0) s_rstd=c10::cuda::compat::rsqrt(val/cols + eps);
            __syncthreads();
            trstd=s_rstd;
            if (j < tcol) 
            {
                #pragma unroll
                for(int ii=0;ii<Vec;ii++){
                int jj=j*Vec+ii;
                intput_vec[ii] = static_cast<T_ACC>(intput_vec[ii]) *trstd* static_cast<T_ACC>(gamma[jj]);
                }
                *(LoadT*)(output+out_idx)=*(LoadT*)intput_vec;
            }
        }
      }
      else
      {
        const int i = blockIdx.x - num_tokens_inputs_embeds;  
        constexpr int share_size=block_size/C10_WARP_SIZE;
        __shared__ T_ACC val_shared[share_size];
        __shared__ T_ACC s_rstd;
        T_ACC val=0;
        int j=threadIdx.x;
        int tcol=cols/Vec;
        int out_cols = cols * 2;
        using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
        scalar_t intput_vec[Vec];
        T_ACC trstd;
        int64_t row_start_offset = i * stride;
        int64_t col_offset = j * Vec;
        int64_t idx = row_start_offset + col_offset;
        int64_t out_idx = i * out_cols + col_offset;
        if (j < tcol) 
        {
            *(LoadT*)intput_vec = *(LoadT*)(previous_data+idx);
            #pragma unroll
            for (int ii = 0; ii < Vec; ii++) {
            val += static_cast<T_ACC>(intput_vec[ii])*static_cast<T_ACC>(intput_vec[ii]);
            }

        }

        val = BlockReduceSum_NEW<T_ACC,block_size>(val,val_shared);
        if (j == 0) s_rstd=c10::cuda::compat::rsqrt(val/cols + eps);
        __syncthreads();
        trstd=s_rstd;
        if (j < tcol) 
        {
            #pragma unroll
            for(int ii=0;ii<Vec;ii++){
            int jj=j*Vec+ii;
            intput_vec[ii] = static_cast<T_ACC>(intput_vec[ii]) *trstd* static_cast<T_ACC>(previous_weight_data[jj]);
            }
            *(LoadT*)(previous_out_data+out_idx)=*(LoadT*)intput_vec;
        }



      }

   
   
   
   
   
   
    }










    
    void fuse_fill_rms_x2_concat(
        torch::Tensor& hidden_states_fuse,
        torch::Tensor& positions, 
        torch::Tensor& inputs_embeds, 
        torch::Tensor& previous_hidden_states, 
        torch::Tensor& weight_inputs_embeds, 
        torch::Tensor& weight_previous_hidden_states, 
        float epsilon) {
    const int hidden_size = inputs_embeds.size(-1);
    const int stride = inputs_embeds.stride(0);
    const int num_tokens_inputs_embeds = inputs_embeds.size(0);
    const int num_tokens_previous_hidden_states = previous_hidden_states.size(0);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 
    auto inp_ptr = reinterpret_cast<std::uintptr_t>(inputs_embeds.data_ptr());
    auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight_inputs_embeds.data_ptr());
    bool ptrs_are_aligned =inp_ptr % 16 == 0  && wt_ptr % 16 == 0;
    const int num_tokens = num_tokens_inputs_embeds + num_tokens_previous_hidden_states;
    
    if(hidden_size%16==0&&hidden_size<=16384&&ptrs_are_aligned){
    AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    inputs_embeds.scalar_type(),
    "fused_rms_norm_contiguous_kernel",
    [&] {
        using T_ACC = at::acc_type<scalar_t, true>;
        T_ACC eps = epsilon;
        scalar_t* self_data  = inputs_embeds.data_ptr<scalar_t>();
        scalar_t* out_data   =  hidden_states_fuse.expect_contiguous()->data_ptr<scalar_t>();
        scalar_t* weight_data= weight_inputs_embeds.expect_contiguous()->data_ptr<scalar_t>();
        int64_t* positions_ptr = positions.data_ptr<int64_t>();

        scalar_t* previous_data  = previous_hidden_states.data_ptr<scalar_t>();
        scalar_t* previous_weight_data= weight_previous_hidden_states.expect_contiguous()->data_ptr<scalar_t>();
        scalar_t* previous_out_data   =  out_data + hidden_size;

        if (hidden_size<=1024){
            fused_fill_rms_x2_concat_kernel<scalar_t,T_ACC,8,128><<<num_tokens,  128, 0, stream>>>(
                positions_ptr,
                self_data,
                out_data,
                weight_data,
                hidden_size,
                eps,
                stride,
                num_tokens_inputs_embeds,
                previous_data,
                previous_out_data,
                previous_weight_data
            );
        }
        else if(hidden_size<=2048){
            fused_fill_rms_x2_concat_kernel<scalar_t,T_ACC,8,256><<<num_tokens,  256, 0, stream>>>(
                positions_ptr,
                self_data,
                out_data,
                weight_data,
                hidden_size,
                eps,
                stride,
                num_tokens_inputs_embeds,
                previous_data,
                previous_out_data,
                previous_weight_data
            );
        }
        else if(hidden_size<=4096){
            if(num_tokens>1200){
                fused_fill_rms_x2_concat_kernel<scalar_t,T_ACC,8,512><<<num_tokens,  512, 0, stream>>>(
                    positions_ptr,
                    self_data,
                    out_data,
                    weight_data,
                    hidden_size,
                    eps,
                    stride,
                    num_tokens_inputs_embeds,
                    previous_data,
                    previous_out_data,
                    previous_weight_data
                );
            }
            else{
                fused_fill_rms_x2_concat_kernel<scalar_t,T_ACC,4,1024><<<num_tokens,  1024, 0, stream>>>(
                    positions_ptr,
                    self_data,
                    out_data,
                    weight_data,
                    hidden_size,
                    eps,
                    stride,
                    num_tokens_inputs_embeds,
                    previous_data,
                    previous_out_data,
                    previous_weight_data
                );
            }
        }
        else if(hidden_size<=8192){
            fused_fill_rms_x2_concat_kernel<scalar_t,T_ACC,8,1024><<<num_tokens,  1024, 0, stream>>>(
                positions_ptr,
                self_data,
                out_data,
                weight_data,
                hidden_size,
                eps,
                stride,
                num_tokens_inputs_embeds,
                previous_data,
                previous_out_data,
                previous_weight_data
            );
        }
        else{
            fused_fill_rms_x2_concat_kernel<scalar_t,T_ACC,16,1024><<<num_tokens,  1024, 0, stream>>>(
                positions_ptr,
                self_data,
                out_data,
                weight_data,
                hidden_size,
                eps,
                stride,
                num_tokens_inputs_embeds,
                previous_data,
                previous_out_data,
                previous_weight_data
            );
        } 



    });    





    }








    

        }

    }
}
