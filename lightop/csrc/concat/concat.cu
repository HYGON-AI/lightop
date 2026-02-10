#include "concat_config.h"
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
// #include "intrinsic.h"
namespace at {
namespace native {


#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define ATOMIC_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      torch::Tensor output_tmp = output.alias();\
      constexpr static bool CONST_NAME = true;  \
      const int BLOCK_SIZE_K_ = 128;            \
      return __VA_ARGS__();                     \
    } else {                                    \
      torch::Tensor output_tmp = output.alias();\
      constexpr static bool CONST_NAME = false; \
      const int BLOCK_SIZE_K_ = 128;            \
      return __VA_ARGS__();                     \
    }                                           \
  }()
  

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))
using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;

template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t per_block_A, 
  uint16_t per_block_B, 
  uint16_t Blcoksize> 
__global__ void __launch_bounds__(1024) concat_prefill_kernel(
  const Element* __restrict__ input_A,
  const Element* __restrict__ input_B,
  const Element* __restrict__ output,
  uint32_t M,
  uint32_t N,
  uint32_t a_section_num, 
  uint32_t b_section_num, 
  uint32_t c_section_num, 
  uint32_t stride_am,
  uint32_t stride_an,
  uint32_t stride_ak, 
  uint32_t stride_bm,
  uint32_t stride_bn,
  uint32_t stride_bk,
  uint32_t block_num_A,
  uint32_t block_num_B
) {
  const int bidx = blockIdx.z; 
  auto ginput_A = const_cast<Element*>(input_A);
  auto ginput_B =  const_cast<Element*>(input_B);
  auto goutput = const_cast<Element*>(output);
  int tid = threadIdx.x;
  int warp_id_vec = tid / 64; 
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); 
  int total_blockidx_num_A = block_num_A /per_block_A;  
  int ele_num_perthread = 16 / sizeof(Element);
  if(bidx < total_blockidx_num_A){ 
  int theradnum_need_by_one_A_section = a_section_num / ele_num_perthread; 
  int num_warp_need_one_section = 0;
  int num_section_perwarp = 0;
  int total_num_section_perblock = 16;
  if(theradnum_need_by_one_A_section >= 64){
    num_warp_need_one_section = theradnum_need_by_one_A_section / 64;
  }
  else{
    num_section_perwarp = 64 / theradnum_need_by_one_A_section;
    total_num_section_perblock = num_section_perwarp * WARP_NUM;
  }
  if(num_section_perwarp > 0 ){
    int tid_insection_A = tid % theradnum_need_by_one_A_section;
    int tid_section_num_A = tid / theradnum_need_by_one_A_section; 
    int itr = (per_block_A / total_num_section_perblock) ==0 ? 1 : (per_block_A / total_num_section_perblock);
    for(int init=0; init< itr ; init++){
      int sectiond = init * total_num_section_perblock + tid_section_num_A ;
      int sub_offset = bidx * per_block_A + sectiond;
      if(sub_offset <= block_num_A - 1){
          auto output_ptr_block_start_base = goutput + sub_offset * c_section_num;
          int M_idx =  sub_offset / N;
          int N_idx =  sub_offset % N;
          auto input_A_ptr_block_start = ginput_A + M_idx * stride_am +  N_idx * stride_an;
          int threadindex_offset = tid_insection_A * ele_num_perthread;
          if( threadindex_offset < a_section_num){
            vec_bf16_8 val_A  =  *(vec_bf16_8*)(input_A_ptr_block_start + threadindex_offset);
            *reinterpret_cast<vec_bf16_8*>(output_ptr_block_start_base + threadindex_offset) = val_A;
          }
      }
    }
  }
  else{
    return ;
  }
}
else{ 
    if(warp_id ==0){
      int theradnum_need_by_one_B_section = b_section_num / ele_num_perthread; 
      int total_blockidx_num_B = block_num_B / per_block_B; 
      int tid_insection_B = tid % theradnum_need_by_one_B_section; 
      int tid_section_num_B = tid / theradnum_need_by_one_B_section; 
      int total_num_section_perblock =  64 / (b_section_num / ele_num_perthread)  * 1;
      int itr = (per_block_B / total_num_section_perblock) ==0 ? 1 : (per_block_B / total_num_section_perblock);  
      for(int init=0; init< itr ; init++){
        int sectiond = init * total_num_section_perblock + tid_section_num_B ;
        int sub_offset_B = (bidx - total_blockidx_num_A) * per_block_B + sectiond;
        if(sub_offset_B <= block_num_B - 1){
            auto output_ptr_block_start_base = goutput + sub_offset_B * c_section_num * N + a_section_num;
            auto input_B_ptr_block_start = ginput_B + sub_offset_B * stride_bm;
            int threadindex_offset = tid_insection_B * ele_num_perthread;
            if( threadindex_offset < b_section_num){
              vec_bf16_8 val_B  =  *(vec_bf16_8*)(input_B_ptr_block_start + threadindex_offset);
              for(int i=0;i<N;i++)
                *reinterpret_cast<vec_bf16_8*>(output_ptr_block_start_base +i * c_section_num + threadindex_offset) = val_B ;
            }
          }
      }
    }
  }
}



template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t per_block_A, 
  uint16_t per_block_B, 
  uint16_t Blcoksize> 
__global__ void __launch_bounds__(1024) concat_decode_kernel(
  const Element* __restrict__ input_A,
  const Element* __restrict__ input_B,
  const Element* __restrict__ output,
  uint32_t M,
  uint32_t N,
  uint32_t a_section_num, 
  uint32_t b_section_num, 
  uint32_t c_section_num, 
  uint32_t stride_am,
  uint32_t stride_an,
  uint32_t stride_ak, 
  uint32_t stride_bm,
  uint32_t stride_bn,
  uint32_t stride_bk,
  uint32_t block_num_A,
  uint32_t block_num_B
) {
  const int bidx = blockIdx.z; 
  auto ginput_A = const_cast<Element*>(input_A);
  auto ginput_B =  const_cast<Element*>(input_B);
  auto goutput = const_cast<Element*>(output);

  int tid = threadIdx.x;
  int warp_id_vec = tid / 64; 
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); 
  
  int total_blockidx_num_A = block_num_A /per_block_A;
  int ele_num_perthread = 16 / sizeof(Element);
//处理A写回的线程块 
if(bidx < total_blockidx_num_A){
  
  int theradnum_need_by_one_A_section = a_section_num / ele_num_perthread; 
  
  int num_warp_need_one_section = 0;
  int num_section_perwarp = 0;
  int total_num_section_perblock = 0;
  
  if(theradnum_need_by_one_A_section >= 64){
    num_warp_need_one_section = theradnum_need_by_one_A_section / 64;
    total_num_section_perblock = WARP_NUM / num_warp_need_one_section;
  }
  else{
    num_section_perwarp = 64 / theradnum_need_by_one_A_section;
    total_num_section_perblock = num_section_perwarp * WARP_NUM;
  }

  if(num_section_perwarp > 0 ){
    int tid_insection_A = tid % theradnum_need_by_one_A_section;
    int tid_section_num_A = tid / theradnum_need_by_one_A_section;
    int itr = (per_block_A / total_num_section_perblock) ==0 ? 1 : (per_block_A / total_num_section_perblock);
    
    for(int init=0; init< itr ; init++){
      int sectiond = init * total_num_section_perblock + tid_section_num_A ;
      int sub_offset = bidx * per_block_A + sectiond;
      if(sub_offset <= block_num_A - 1){
          auto output_ptr_block_start_base = goutput + sub_offset * c_section_num;
          int M_idx =  sub_offset / N;
          int N_idx =  sub_offset % N;
          auto input_A_ptr_block_start = ginput_A + M_idx * stride_am +  N_idx * stride_an;
          int threadindex_offset = tid_insection_A * ele_num_perthread;
          if( threadindex_offset < a_section_num){
            vec_bf16_8 val_A  =  *(vec_bf16_8*)(input_A_ptr_block_start + threadindex_offset);
            *reinterpret_cast<vec_bf16_8*>(output_ptr_block_start_base + threadindex_offset) = val_A;
          }
      }
    }
  }
   
  else if (num_warp_need_one_section > 0 ){

    int tid_insection_A = tid % theradnum_need_by_one_A_section;

    int tid_section_num_A = tid / theradnum_need_by_one_A_section; //0~3
    
    int itr = (per_block_A / total_num_section_perblock) ==0 ? 1 : (per_block_A / total_num_section_perblock);
    for(int init=0; init< itr ; init++){
     
      int sectiond = init * total_num_section_perblock + tid_section_num_A ;

      int sub_offset = bidx * per_block_A + sectiond;

      if(sub_offset <= block_num_A - 1){
          
        auto output_ptr_block_start_base = goutput + sub_offset * c_section_num;
        
          int M_idx =  sub_offset / N;
          int N_idx =  sub_offset % N;
        
          auto input_A_ptr_block_start = ginput_A + M_idx * stride_am +  N_idx * stride_an;
          int threadindex_offset = tid_insection_A * 8;
          if( threadindex_offset < a_section_num){
            vec_bf16_8 val_A  =  *(vec_bf16_8*)(input_A_ptr_block_start + threadindex_offset);
            *reinterpret_cast<vec_bf16_8*>(output_ptr_block_start_base + threadindex_offset) = val_A;
          }
      }
    }
  }
  else{
    return ;
  }
}
//处理B写回的线程块
else{ 
 
    if(warp_id ==0){
      
      int theradnum_need_by_one_B_section = b_section_num / ele_num_perthread; 
      
      int total_blockidx_num_B = block_num_B / per_block_B;

      int tid_insection_B = tid % theradnum_need_by_one_B_section;

      int tid_section_num_B = tid / theradnum_need_by_one_B_section;

      int total_num_section_perblock =  64 / theradnum_need_by_one_B_section;

      int itr = (per_block_B / total_num_section_perblock) ==0 ? 1 : (per_block_B / total_num_section_perblock);  
      for(int init=0; init< itr ; init++){

        int sectiond = init * total_num_section_perblock + tid_section_num_B ;

        int sub_offset_B = (bidx - total_blockidx_num_A) * per_block_B + sectiond;

        if(sub_offset_B <= block_num_B - 1){

            auto output_ptr_block_start_base = goutput + sub_offset_B * c_section_num + a_section_num;
            int M_idx =  sub_offset_B / N;
            int N_idx =  sub_offset_B % N; 
            auto input_B_ptr_block_start = ginput_B + M_idx * stride_bm + N_idx *stride_bn;
            int threadindex_offset = tid_insection_B * ele_num_perthread;
            if( threadindex_offset < b_section_num){
              vec_bf16_8 val_B  =  *(vec_bf16_8*)(input_B_ptr_block_start + threadindex_offset);
                *reinterpret_cast<vec_bf16_8*>(output_ptr_block_start_base + threadindex_offset) = val_B ;
            }
          }
      }
    }
  }
}


template<int per_block_A, int per_block_B, int Blcoksize,int WARP_NUM , typename T>
void launch_concat_decode(const GemmParams<T>& params) {
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64; 
    blockDim.y = 1;
    blockDim.z = 1;
    
    gridDim.z = DIVIDE(params.M * params.N, per_block_A) + DIVIDE(params.M * params.N, per_block_B);
    int block_num_B = params.M * params.N ;
    gridDim.y = 1; 
    gridDim.x = 1; 
    constexpr int lds_size = 0;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    int block_num_A = params.M * params.N;
    concat_decode_kernel<T, WARP_NUM, per_block_A, per_block_B,Blcoksize><<<gridDim, blockDim, lds_size, stream>>>(
      params.input_A, 
      params.input_B, 
      params.ptr_C, 
      params.M,
      params.N,
      params.a_section_num,
      params.b_section_num,
      params.c_section_num,
      params.stride_am,
      params.stride_an,
      params.stride_ak,
      params.stride_bm,
      params.stride_bn,
      params.stride_bk,
      block_num_A,
      block_num_B
      );

      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in concat: %s\n", cudaGetErrorString(err));
      }
} 


template<int per_block_A, int per_block_B, int Blcoksize,int WARP_NUM , typename T>
void launch_concat_prefill(const GemmParams<T>& params) {
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM*64; 
    blockDim.y = 1;
    blockDim.z = 1;
    //针对deepseek inputB stride2为0的特定优化
    gridDim.z = DIVIDE(params.M * params.N, per_block_A) + DIVIDE(params.M * params.N, per_block_B);
    int block_num_B = params.M * params.N ;
    if(params.stride_bn == 0){

      gridDim.z = DIVIDE(params.M * params.N, per_block_A) + DIVIDE(params.M, per_block_B);
      block_num_B = params.M;
    }
    gridDim.y = 1; 
    gridDim.x = 1; 
    constexpr int lds_size = 0;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
    int block_num_A = params.M * params.N;
    concat_prefill_kernel<T, WARP_NUM, per_block_A, per_block_B,Blcoksize><<<gridDim, blockDim, lds_size, stream>>>(
      params.input_A, 
      params.input_B, 
      params.ptr_C, 
      params.M,
      params.N,
      params.a_section_num,
      params.b_section_num,
      params.c_section_num,
      params.stride_am,
      params.stride_an,
      params.stride_ak,
      params.stride_bm,
      params.stride_bn,
      params.stride_bk,
      block_num_A,
      block_num_B
      );

      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("CUDA error in concat: %s\n", cudaGetErrorString(err));
      }
}  



torch::Tensor ds_concat(
  torch::Tensor input_A,
  torch::Tensor input_B,
  torch::Tensor output,
  int32_t mode
  ){
    if(input_A.dim()==3 && input_B.dim()==3 && output.dim()==3 ){
      int64_t M = input_A.size(0);
      int64_t N = input_A.size(1);
      int a_section_num = input_A.size(2);
      int b_section_num = input_B.size(2);
      int c_section_num = output.size(2);
      const int stride_am = input_A.stride(0); 
      const int stride_an = input_A.stride(1); 
      const int stride_ak = input_A.stride(2); 
      const int stride_bm = input_B.stride(0); 
      const int stride_bn = input_B.stride(1); 
      const int stride_bk = input_B.stride(2); 
      if (input_A.scalar_type() == at::ScalarType::BFloat16){
        GemmParams<bhalf_t> params_in(
          (bhalf_t*)input_A.data_ptr(), 
          (bhalf_t*)input_B.data_ptr(), 
          (bhalf_t*)output.data_ptr(),
          M,
          N,
          a_section_num,
          b_section_num,
          c_section_num,
          stride_am,
          stride_an,
          stride_ak,
          stride_bm,
          stride_bn,
          stride_bk
        );
        if(M < 1024){
          auto it = kernel_maps_decode.find(mode);
          if (it != kernel_maps_decode.end()) {
            it->second(params_in);
          } 
          else {
              printf("No matching kernel configuration found, using default settings \n");
              launch_concat_decode<4, 8, 1024, 4, bhalf_t>(params_in);
          }
        }else{ 
          auto it = kernel_maps_prefill.find(mode);
          if (it != kernel_maps_prefill.end()) {
            it->second(params_in);
          } 
          else {
              printf("No matching kernel configuration found, using default settings \n");
              launch_concat_prefill<16, 8, 1024, 1, bhalf_t>(params_in);
          }
        }
      }
      else if (input_A.scalar_type() == at::ScalarType::Half){
        GemmParams<half> params_in(
          (half*)input_A.data_ptr(), 
          (half*)input_B.data_ptr(), 
          (half*)output.data_ptr(),
          M,
          N,
          a_section_num,
          b_section_num,
          c_section_num,
          stride_am,
          stride_an,
          stride_ak,
          stride_bm,
          stride_bn,
          stride_bk
        );
        if(M < 1024){
          auto it = kernel_maps_decode_half.find(mode);
          if (it != kernel_maps_decode_half.end()) {
            it->second(params_in);
          } 
          else {
              printf("No matching kernel configuration found, using default settings \n");
              launch_concat_decode<4, 8, 1024, 4, half>(params_in);
          }
        }else{ 
          auto it = kernel_maps_prefill_half.find(mode);
          if (it != kernel_maps_prefill_half.end()) {
            it->second(params_in);
          } 
          else {
              printf("No matching kernel configuration found, using default settings \n");
              launch_concat_prefill<16, 8, 1024, 1, half>(params_in);
          }
        }        
      }
      else{
        GemmParams<uint8_t> params_in(
          (uint8_t*)input_A.data_ptr(), 
          (uint8_t*)input_B.data_ptr(), 
          (uint8_t*)output.data_ptr(),
          M,
          N,
          a_section_num,
          b_section_num,
          c_section_num,
          stride_am,
          stride_an,
          stride_ak,
          stride_bm,
          stride_bn,
          stride_bk
        );
        if(M < 1024){
          auto it = kernel_maps_decode_fp8.find(mode);
          if (it != kernel_maps_decode_fp8.end()) {
            it->second(params_in);
          } 
          else {
              printf("No matching kernel configuration found, using default settings \n");
              launch_concat_decode<4, 8, 1024, 4, uint8_t>(params_in);
          }
        }else{ 
          auto it = kernel_maps_prefill_fp8.find(mode);
          if (it != kernel_maps_prefill_fp8.end()) {
            it->second(params_in);
          } 
          else {
              printf("No matching kernel configuration found, using default settings \n");
              launch_concat_prefill<16, 8, 1024, 1, uint8_t>(params_in);
          }
        }        
      }
    }
    else{
      printf("input tensor must be dim 3\n");
    }        
    return output;
}

}
}

