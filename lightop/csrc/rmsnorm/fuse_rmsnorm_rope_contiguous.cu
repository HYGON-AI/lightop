// #include "type_convert.cuh"
// #include "dispatch_utils.h"

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/AccumulateType.h>
#include <THC/THCDeviceUtils.cuh>
#include <ATen/Dispatch.h>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
template <typename T, int N> struct alignas(sizeof(T) * N) Vector {
  T val[N];
};

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

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


template <typename scalar_t,typename T_ACC, int Vec=8, int num_warp_per_block=4, int rot_dim=64, int kv_c_dim=512>
__global__ void rms_rope_kernel(
    scalar_t* __restrict__ input_q,  // [bs, num_head, q_c_dim + rot_dim]
    scalar_t* __restrict__ input_k,  // [bs, kv_c_dim + rot_dim]
    scalar_t* __restrict__ input_kv_c,  // [bs, kv_c_dim]
    scalar_t* __restrict__ weight,  // [kv_c_dim]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim / 2]
    const int64_t* __restrict__ positions,  // [bs]
    const int q_c_dim,  // 128
    const int num_head,  // 128
    const int q_stride_head,
    const int k_stride_head,
    const int q_grid_x,
    const int k_grid_x,
    const int k_rms_grid_x,
    const int bs,
    const float epsilon) 
  {

    if(blockIdx.x < q_grid_x){
      const int num_head_per_warp = (C10_WARP_SIZE * Vec) / rot_dim;
      const int num_head_per_block = num_head_per_warp * num_warp_per_block;
      const int block_per_token = (num_head + num_head_per_block - 1) / num_head_per_block;
      using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
      using LoadTheta = at::native::memory::aligned_vector<scalar_t, Vec/2>;
      const int embed_dim = rot_dim / 2;
      const int laneid = threadIdx.x % C10_WARP_SIZE;
      const int warpid = threadIdx.x / C10_WARP_SIZE;
      const int thread_per_head = rot_dim / Vec;
      // 计算当前处理的token
      const int token_idx = blockIdx.x / block_per_token;
      if (token_idx >= bs) return;
      // if(blockIdx.x == 0 && threadIdx.x == 0){
      //   printf("bs: %d, num_head: %d, q_c_dim: %d, rot_dim: %d, q_stride_head: %d, token_idx: %d, block_per_token: %d\n", 
      //     bs, num_head, q_c_dim, rot_dim, q_stride_head, token_idx, block_per_token);
      // }
      // 计算当前处理的head块
      const int head_block_idx = blockIdx.x % block_per_token;     // 0-3
      const int head_start = head_block_idx * num_head_per_block;  // 0, 32, 64, 96
      const int lane_of_head = laneid / thread_per_head;  // 0, 1, 2, 3
      const int head_in_block = warpid * num_head_per_warp + lane_of_head;  // 0, 8, 16, 24
      const int64_t pos = positions[token_idx];
      const int head_offset_in_token = head_start + head_in_block;
      if(head_offset_in_token >= num_head) return;
      // 计算q_pe的起始地址， 并读取q_pe
      const int element_in_head = (threadIdx.x % thread_per_head)*Vec; // 0 8 16 24 32 40 48 56 
      const int q_pe_offset = token_idx * num_head * (q_stride_head) + head_offset_in_token * (q_stride_head) + element_in_head;
      scalar_t* q_pe_ptr = input_q + q_pe_offset;
      // 计算cos_sin地址，并读取cos_sin
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
      const int cos_index_in_cache = element_in_head /2;
      const int sin_index_in_cache = cos_index_in_cache + embed_dim;
      // 读取q_pe、cos、sin
      scalar_t input_vec[Vec];
      scalar_t cos_vec[Vec/2];
      scalar_t sin_vec[Vec/2];
      *(LoadT*)(input_vec) = *(LoadT*)(q_pe_ptr);
      *(LoadTheta*)cos_vec = *(LoadTheta*)(cache_ptr + cos_index_in_cache);
      *(LoadTheta*)sin_vec = *(LoadTheta*)(cache_ptr + sin_index_in_cache);
      for(int ii = 0, rot_dim_idx = 0; ii < Vec; ii+=2){
        scalar_t x = input_vec[ii];
        scalar_t y = input_vec[ii+1];
        scalar_t cos_val = cos_vec[rot_dim_idx];
        scalar_t sin_val = sin_vec[rot_dim_idx];
        input_vec[ii] = x * cos_val - y * sin_val;
        input_vec[ii+1] = y * cos_val + x * sin_val;
        // if(blockIdx.x == 0 && threadIdx.x == 0){
        //   printf("x: %f, y: %f, cos_val: %f, sin_val: %f, input_vec[ii]: %f, input_vec[ii+1]: %f, token_idx: %d, pos: %d, head_block_idx: %d, head_in_block: %d, lane_of_head: %d, element_in_head: %d, q_c_dim: %d, head_start: %d, q_pe_offset: %d\n", 
        //     float(x), float(y), float(cos_val), float(sin_val), float(input_vec[ii]), float(input_vec[ii+1]), token_idx, int(pos), head_block_idx, head_in_block, lane_of_head, element_in_head, q_c_dim, head_start, q_pe_offset);
        // }
        rot_dim_idx += 1;
      }
      // 写回q_pe
      *(LoadT*)q_pe_ptr = *(LoadT*)input_vec;
    }
    else if(blockIdx.x < q_grid_x+k_grid_x){
      constexpr int thread_per_head = rot_dim / Vec;
      constexpr int element_per_warp = C10_WARP_SIZE * Vec;
      constexpr int num_token_per_warp = (element_per_warp) / rot_dim;
      constexpr int num_token_per_block = num_token_per_warp * num_warp_per_block;
      using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
      using LoadTheta = at::native::memory::aligned_vector<scalar_t, Vec/2>;
      constexpr int embed_dim = rot_dim / 2;
      const int warpid = threadIdx.x / C10_WARP_SIZE;
      const int laneid = threadIdx.x % C10_WARP_SIZE;
      const int token_idx = (blockIdx.x - q_grid_x)*num_token_per_block + warpid*num_token_per_warp + laneid/thread_per_head;
      if(token_idx >= bs) return;
      // 计算k_pe的offset
      const int thread_offset_in_token = (laneid % thread_per_head)*Vec;
      const int k_pe_offset = token_idx * k_stride_head + thread_offset_in_token;
      scalar_t* k_pe_ptr = input_k + k_pe_offset;
      // 计算cos_sin的偏移量
      const int64_t pos = positions[token_idx];
      const int cos_index_in_cache = thread_offset_in_token /2;
      const int sin_index_in_cache = cos_index_in_cache + embed_dim;
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
      // 读取q_pe、cos、sin
      scalar_t input_vec[Vec];
      scalar_t cos_vec[Vec/2];
      scalar_t sin_vec[Vec/2];
      *(LoadT*)(input_vec) = *(LoadT*)(k_pe_ptr);
      *(LoadTheta*)cos_vec = *(LoadTheta*)(cache_ptr + cos_index_in_cache);
      *(LoadTheta*)sin_vec = *(LoadTheta*)(cache_ptr + sin_index_in_cache);

      for(int ii = 0, rot_dim_idx = 0; ii < Vec; ii+=2){
        scalar_t x = input_vec[ii];
        scalar_t y = input_vec[ii+1];
        scalar_t cos_val = cos_vec[rot_dim_idx];
        scalar_t sin_val = sin_vec[rot_dim_idx];
        input_vec[ii] = x * cos_val - y * sin_val;
        input_vec[ii+1] = y * cos_val + x * sin_val;
        rot_dim_idx += 1;
      }
      // 写回k_pe
      *(LoadT*)k_pe_ptr = *(LoadT*)input_vec;
    }
    else
    {
      constexpr int element_per_warp = C10_WARP_SIZE * Vec; // 512
      constexpr int token_per_warp = element_per_warp / (kv_c_dim); // 512 / 512 = 1
      constexpr int token_per_block = token_per_warp * num_warp_per_block; // 1 * 4 = 4
      using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
      using LoadTheta = at::native::memory::aligned_vector<scalar_t, Vec/2>;
      const int warpid = threadIdx.x / C10_WARP_SIZE;
      const int laneid = threadIdx.x % C10_WARP_SIZE;
      const int block_token_id = (blockIdx.x - q_grid_x - k_grid_x) * token_per_block;
      const int warp_token_id = warpid * token_per_warp;
      const int token_id = (blockIdx.x - q_grid_x - k_grid_x) * token_per_block + warp_token_id; // 每个block负责4个token，每个warp负责1个token
      if(token_id >= bs) return;
      // 计算kv_c的offset
      const int thread_offset_in_token = laneid*Vec;
      if(thread_offset_in_token >= kv_c_dim) return;
      const int kv_c_offset = token_id * k_stride_head + thread_offset_in_token;
      scalar_t* kv_c_ptr = input_kv_c + kv_c_offset;
      // 计算weight的offset
      scalar_t* weight_ptr = weight + thread_offset_in_token;
      // 读取kv_c, weight
      scalar_t input_vec[Vec];
      scalar_t weight_vec[Vec];
      *(LoadT*)(input_vec) = *(LoadT*)(kv_c_ptr);
      *(LoadT*)(weight_vec) = *(LoadT*)(weight_ptr);
      T_ACC val = 0;
      T_ACC s_rstd;
      #pragma unroll
      for (int ii = 0; ii < Vec; ii++) {
        val += static_cast<T_ACC>(input_vec[ii]) * static_cast<T_ACC>(input_vec[ii]);
      }
      val = BlockReduceSum_NEW<T_ACC,C10_WARP_SIZE>(val,nullptr);
      if(laneid == 0) {
        s_rstd = c10::cuda::compat::rsqrt(val / kv_c_dim + epsilon);
      }
      s_rstd = __shfl(s_rstd, 0);
      #pragma unroll
      for(int ii = 0; ii < Vec; ii+=1){
        input_vec[ii] = static_cast<T_ACC>(input_vec[ii]) * s_rstd * static_cast<T_ACC>(weight_vec[ii]);
      }
      *(LoadT*)kv_c_ptr = *(LoadT*)input_vec;
    }
  }



  void fused_rms_norm_rope_contiguous(torch::Tensor& positions, torch::Tensor& q_pe, torch::Tensor& k_pe, torch::Tensor& kv_c, 
    torch::Tensor& weight, torch::Tensor& cos_sin_cache, bool is_neox_style, float epsilon) {

    // printf("hello fused_rms_norm_rope_contiguous\n");
    // 暂时只实现rope逻辑，rms norm逻辑后续添加
    // const int bs = q_pe.size(0);
    // constexpr int q_c_dim = 128;
    // constexpr int rot_dim = 64;
    // const int num_head = num_head_q;
    TORCH_CHECK(q_pe.dim() == 3, "q_pe must be 3D tensor [bs, num_head, rot_dim]");
    const int bs = q_pe.size(0);
    const int num_head = q_pe.size(1);
    constexpr int q_c_dim = 128;
    constexpr int rot_dim = 64;
    const int q_stride_head = q_pe.stride(1);
    TORCH_CHECK(q_stride_head == q_c_dim+rot_dim, "q_stride_head must be q_c_dim + rot_dim");
    TORCH_CHECK(q_pe.size(2) ==  rot_dim, "q_pe.size(2) must be rot_dim");
    TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must be 2D tensor [max_position, rot_dim]");
    TORCH_CHECK(cos_sin_cache.size(1) == rot_dim, "cos_sin_cache.size(1) must equal rot_dim");

    // 计算ROPE(Q)grid和block尺寸
    constexpr int num_warp_per_block = 4;
    constexpr int Vec = 8;
    constexpr int num_head_per_warp = (C10_WARP_SIZE * Vec) / rot_dim;  // 512 / 64 = 8
    constexpr int num_head_per_block = num_head_per_warp * num_warp_per_block;  // 8 * 4 = 32
    const int block_per_token = (num_head + num_head_per_block - 1) / num_head_per_block;  // 向上取整
    const int q_grid_x = block_per_token * bs;  // 4 * bs
    const int block_x = num_warp_per_block * C10_WARP_SIZE;  // 128
    // 计算ROPE(K)grid和block尺寸
    constexpr int kv_c_dim = 512;
    constexpr int k_stride_head = kv_c_dim + rot_dim;
    TORCH_CHECK(k_pe.dim() == 2, "k_pe must be 2D tensor [bs, kv_c_dim + rot_dim]");
    TORCH_CHECK(k_pe.size(1) ==  rot_dim, "k_pe.size(1) must be rot_dim");
    TORCH_CHECK(k_pe.size(0) == bs, "k_pe.size(0) must be bs");
    TORCH_CHECK(k_pe.stride(0) == kv_c_dim + rot_dim, "k_pe.stride(0) must be kv_c_dim + rot_dim");
    
    constexpr int element_per_warp = C10_WARP_SIZE * Vec;
    constexpr int k_token_per_warp = element_per_warp / (rot_dim);  // 8
    constexpr int k_token_per_block = k_token_per_warp * num_warp_per_block; // 32
    const int k_grid_x = (bs + k_token_per_block - 1) / k_token_per_block; // (32 + 32 - 1) / 32 = 1
    
    // 计算RMS(K)grid和block尺寸
    TORCH_CHECK(kv_c.dim() == 2, "kv_c must be 2D tensor [bs, kv_c_dim + rot_dim]");
    TORCH_CHECK(kv_c.size(1) == kv_c_dim, "kv_c.size(1) must be kv_c_dim");
    TORCH_CHECK(kv_c.size(0) == bs, "kv_c.size(0) must be bs");
    TORCH_CHECK(kv_c.stride(0) == kv_c_dim + rot_dim, "kv_c.stride(0) must be kv_c_dim + rot_dim");
    constexpr int k_rms_token_per_warp = element_per_warp / (kv_c_dim);// 1
    constexpr int k_rms_token_per_block = k_rms_token_per_warp * num_warp_per_block; // 4
    const int k_rms_grid_x = (bs + k_rms_token_per_block - 1) / k_rms_token_per_block; // (32 + 32 - 1) / 32 = 1





    const int grid_x = q_grid_x + k_grid_x + k_rms_grid_x;
    // const int grid_x = q_grid_x + k_grid_x;
    dim3 grid(grid_x);
    dim3 block(block_x);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q_pe));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int temp1;

    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q_pe.scalar_type(),
      "fused_rms_norm_contiguous_kernel",
      [&]{
        using T_ACC = at::acc_type<scalar_t, true>;
        scalar_t* q_pe_ptr  = q_pe.data_ptr<scalar_t>();
        scalar_t* k_pe_ptr  = k_pe.data_ptr<scalar_t>();
        scalar_t* kv_c_ptr = kv_c.data_ptr<scalar_t>();
        scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
        const scalar_t* cos_sin_cache_ptr = cos_sin_cache.data_ptr<scalar_t>();
        const int64_t* positions_ptr = positions.data_ptr<int64_t>();

        rms_rope_kernel<scalar_t,T_ACC,Vec,num_warp_per_block,rot_dim,kv_c_dim><<<grid, block, 0, stream>>>(
            q_pe_ptr,k_pe_ptr,kv_c_ptr,weight_ptr,cos_sin_cache_ptr,positions_ptr,q_c_dim,num_head,q_stride_head,k_stride_head,q_grid_x,k_grid_x, k_rms_grid_x, bs, epsilon);

      });

}

}
}