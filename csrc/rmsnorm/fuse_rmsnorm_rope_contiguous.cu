// #include "type_convert.cuh"
// #include "dispatch_utils.h"

#include <torch/all.h>
#include <cmath>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/AccumulateType.h>
#include <THC/THCDeviceUtils.cuh>
#include <ATen/Dispatch.h>
#include "hip/hip_fp16.h"
#include "hip/hip_bf16.h"
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
template <typename T, int N> struct alignas(sizeof(T) * N) Vector {
  T val[N];
};

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

#ifdef TORCH_VERSION_OVER_290
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif
namespace at{
namespace native{

using bhalf_t = __hip_bfloat16;

static __device__ inline float clamp_rms_rope_value(float value, float limit) {
  return fminf(fmaxf(value, -limit), limit);
}

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)


  inline __device__ uint8_t float_to_fp8e5m2(float f) {
    constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
    constexpr uint32_t fp8_max = UINT32_C(143) << 23;
    constexpr uint32_t denorm_mask = UINT32_C(134) << 23;
    uint32_t f_bits = c10::detail::fp32_to_bits(f);
    uint8_t result = 0u;
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
      result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
    } else {
      if (f_bits < (UINT32_C(113) << 23)) {
        f_bits = c10::detail::fp32_to_bits(c10::detail::fp32_from_bits(f_bits)
                 + c10::detail::fp32_from_bits(denorm_mask));
        result = static_cast<uint8_t>(f_bits - denorm_mask);
      } else {
        uint32_t mant_odd = (f_bits >> 21) & 1;
        f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;
        f_bits += mant_odd;
        result = static_cast<uint8_t>(f_bits >> 21);
      }
    }
    result |= static_cast<uint8_t>(sign >> 24);
    return result;
  }


  inline __device__ uint8_t float_to_fp8e4m3(float f) {
    // ===== 常量定义 =====
    // FP8 E4M3最大值: 1087 << 20
    // 1087 = 2^(4+1) - 1 + 2^4 = 31 + 16 = 47? 等等，这里需要重新计算
    // 实际上是基于指数和尾数的位模式
    constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
    
    // 非正规化数处理掩码
    constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
    
    // ===== 步骤1: 提取float的位表示 =====
    uint32_t f_bits = c10::detail::fp32_to_bits(f);
    uint8_t result = 0u;
    
    // ===== 步骤2: 提取符号位并移除 =====
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;
    
    // ===== 步骤3: 处理不同数值范围 =====
    
    // 3a. 溢出处理
    if (f_bits >= fp8_max) {
      result = 0x7f;  // 01111111 (NaN/最大值)
    } else {
      
      // 3b. 非正规化数处理
      if (f_bits < (UINT32_C(121) << 23)) {
        f_bits = c10::detail::fp32_to_bits(
          c10::detail::fp32_from_bits(f_bits) + 
          c10::detail::fp32_from_bits(denorm_mask)
        );
        result = static_cast<uint8_t>(f_bits - denorm_mask);
      
      // 3c. 正规化数处理
      } else {
        // 检查尾数第20位是否为1（对应FP8 E4M3 3位尾数的LSB）
        uint8_t mant_odd = (f_bits >> 20) & 1;
        
        // 指数偏置调整: FP32偏置127 → FP8 E4M3偏置7
        // 调整量 = 7 - 127 = -120
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        
        // 应用奇数舍入
        f_bits += mant_odd;
        
        // 右移20位得到8位结果
        result = static_cast<uint8_t>(f_bits >> 20);
      }
    }
    
    // ===== 步骤4: 加上符号位 =====
    result |= static_cast<uint8_t>(sign >> 24);
    
    return result;
  }




  inline __device__ uint32_t float_to_fp8e4m3_x4(float a, float b, float c, float d)
  {
    #if defined(__gfx938__)
      int32_t res;
      res = __builtin_hcu_cvt_pk_fp8_f32(a, b, res, false);
      res = __builtin_hcu_cvt_pk_fp8_f32(c, d, res, true);
      return res;
    #else
      int32_t out;
      reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e4m3(a);
      reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e4m3(b);
      reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e4m3(c);
      reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e4m3(d);
      return out;
    #endif
  }
  
  inline __device__ float float_to_bf16_to_float(float f) {
    #if defined(__gfx938__)
        // HCU gfx938 原生硬件 bf16 转换（最优）
        return float(bhalf_t(f));
    #else
        // 通用架构：你提供的软件实现
        uint32_t u = *(uint32_t*)(&f);
        u += 0x7fff + ((u >> 16) & 1);
        uint32_t bf16_u = u >> 16;
        uint32_t fp32_u = bf16_u << 16;
        return *(float*)(&fp32_u);
    #endif
    }

  // 主函数：4 x float → bf16 → float → fp8e4m3 → 打包 uint32_t
inline __device__ uint32_t float_to_bhalf_to_fp8e4m3_x4(float a, float b, float c, float d)
{
    // 统一先做 bf16 中介转换
    float a_bf = float_to_bf16_to_float(a);
    float b_bf = float_to_bf16_to_float(b);
    float c_bf = float_to_bf16_to_float(c);
    float d_bf = float_to_bf16_to_float(d);

    #if defined(__gfx938__)
      int32_t res;
      res = __builtin_hcu_cvt_pk_fp8_f32(a_bf, b_bf, res, false);
      res = __builtin_hcu_cvt_pk_fp8_f32(c_bf, d_bf, res, true);
      return (uint32_t)res;
    #else
      int32_t out = 0;
      reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e4m3(a_bf);
      reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e4m3(b_bf);
      reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e4m3(c_bf);
      reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e4m3(d_bf);
      return (uint32_t)out;
    #endif
}
  
  
  inline __device__ uint32_t float_to_fp8e5m2_x4(float a, float b, float c, float d)
  {
    #if defined(__gfx938__)
      int32_t res;
      res = __builtin_hcu_cvt_pk_bf8_f32(a, b, res, false);
      res = __builtin_hcu_cvt_pk_bf8_f32(c, d, res, true);
      return res;
    #else
      int32_t out;
      reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e5m2(a);
      reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e5m2(b);
      reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e5m2(c);
      reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e5m2(d);
      return out;
    #endif
  }





// 主函数：4 x float → bf16 → float → fp8e5m2 → 打包 uint32_t
inline __device__ uint32_t float_to_bhalf_to_fp8e5m2_x4(float a, float b, float c, float d)
{
    // 统一流程：所有输入先执行 float → bf16 → float
    float a_bf = float_to_bf16_to_float(a);
    float b_bf = float_to_bf16_to_float(b);
    float c_bf = float_to_bf16_to_float(c);
    float d_bf = float_to_bf16_to_float(d);

#if defined(__gfx938__)
    // gfx938 硬件批量转换 + 打包 fp8e5m2
    int32_t res;
    res = __builtin_hcu_cvt_pk_bf8_f32(a_bf, b_bf, res, false);
    res = __builtin_hcu_cvt_pk_bf8_f32(c_bf, d_bf, res, true);
    return (uint32_t)res;
#else
    // 通用架构：逐个转 fp8e5m2 并按字节打包
    int32_t out = 0;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e5m2(a_bf);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e5m2(b_bf);
    reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e5m2(c_bf);
    reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e5m2(d_bf);
    return (uint32_t)out;
#endif
}



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


template <typename scalar_t,typename T_ACC, int Vec=8, int num_warp_per_block=4, int rot_dim=64,
int kv_c_dim=512, bool APPLY_CLAMP=false>
__global__ void rms_rope_kernel(
    scalar_t* __restrict__ input_q,  // [bs, num_head, q_c_dim + rot_dim]
    scalar_t* __restrict__ input_k,  // [bs, kv_c_dim + rot_dim]
    scalar_t* __restrict__ input_kv_c,  // [bs, kv_c_dim]
    scalar_t* __restrict__ output_kv_c_normed,  // [bs, kv_c_dim]
    scalar_t* __restrict__ weight,  // [kv_c_dim]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim / 2]
    const int64_t* __restrict__ positions,  // [bs]
    const int64_t* __restrict__ slot_mapping,  // [bs]
    void* __restrict__ kv_cache,  // [num_kv_cache_blocks, kv_cache_blocks_size, kv_c_dim + rot_dim]
    const int kv_cache_blocks_size,
    const int q_c_dim,  // 128
    const int num_head,  // 128
    const int q_stride_head,
    const int k_stride_head,
    const int q_grid_x,
    const int k_grid_x,
    const int k_rms_grid_x,
    const int bs,
    const float epsilon,
    const float limit,
    const int kv_dtype,
    const int max_slot_id,
    const int k_bs_stride) 
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
      if(head_offset_in_token >= num_head) return;
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
      using OutputT = at::native::memory::aligned_vector<uint8_t, Vec>;
      constexpr int embed_dim = rot_dim / 2;
      const int warpid = threadIdx.x / C10_WARP_SIZE;
      const int laneid = threadIdx.x % C10_WARP_SIZE;
      const int token_idx =  (blockIdx.x - q_grid_x)*num_token_per_block + warpid*num_token_per_warp + laneid/thread_per_head;
      if(token_idx >= bs) return;  
      
      // 计算k_pe的offset
      const int thread_offset_in_token = (laneid % thread_per_head)*Vec;
      // const int k_pe_offset = token_idx * k_stride_head + thread_offset_in_token;
      const int k_pe_offset = token_idx * k_bs_stride + thread_offset_in_token;
      scalar_t* k_pe_ptr = input_k + k_pe_offset;
      // 计算cos_sin的偏移量
      const int64_t pos = positions[token_idx];
      const int cos_index_in_cache = thread_offset_in_token /2;
      const int sin_index_in_cache = cos_index_in_cache + embed_dim;
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
      // 计算output的偏移量
      const int64_t slot = slot_mapping[token_idx];
      if(slot >= max_slot_id || slot < 0)
      {
        return;
      }



      const int64_t kv_cache_offset = slot * 576 + thread_offset_in_token;
      // 读取q_pe、cos、sin
      scalar_t input_vec[Vec];
      scalar_t cos_vec[Vec/2];
      scalar_t sin_vec[Vec/2];
      uint8_t output_vec[Vec];
      float float_vec[Vec];
      *(LoadT*)(input_vec) = *(LoadT*)(k_pe_ptr);
      *(LoadTheta*)cos_vec = *(LoadTheta*)(cache_ptr + cos_index_in_cache);
      *(LoadTheta*)sin_vec = *(LoadTheta*)(cache_ptr + sin_index_in_cache);

      if (kv_dtype != 2) // fp8_e5m2 or fp8_e4m3
      {
        for(int ii = 0, rot_dim_idx = 0; ii < Vec; ii+=2){
          scalar_t x = input_vec[ii];
          scalar_t y = input_vec[ii+1];
          scalar_t cos_val = cos_vec[rot_dim_idx];
          scalar_t sin_val = sin_vec[rot_dim_idx];
          // output_vec[ii] = float_to_fp8e5m2(x * cos_val - y * sin_val);
          // output_vec[ii+1] = float_to_fp8e5m2(y * cos_val + x * sin_val);
          float_vec[ii] = x * cos_val - y * sin_val;
          float_vec[ii+1] = y * cos_val + x * sin_val;
          input_vec[ii] = static_cast<scalar_t>(float_vec[ii]);
          input_vec[ii+1] = static_cast<scalar_t>(float_vec[ii+1]);
          rot_dim_idx += 1;
        }
        if(kv_dtype == 0)
        {
          ((uint32_t*)output_vec)[0] = float_to_bhalf_to_fp8e5m2_x4(float_vec[0], float_vec[1], float_vec[2], float_vec[3]);
          ((uint32_t*)output_vec)[1] = float_to_bhalf_to_fp8e5m2_x4(float_vec[4], float_vec[5], float_vec[6], float_vec[7]);
        }
        else
        {
          ((uint32_t*)output_vec)[0] = float_to_bhalf_to_fp8e4m3_x4(float_vec[0], float_vec[1], float_vec[2], float_vec[3]);
          ((uint32_t*)output_vec)[1] = float_to_bhalf_to_fp8e4m3_x4(float_vec[4], float_vec[5], float_vec[6], float_vec[7]);
        }
        uint8_t* output_ptr = (uint8_t*)kv_cache + kv_cache_offset + 512;
        *(OutputT*)output_ptr = *(OutputT*)output_vec;
        *(LoadT*)(k_pe_ptr) = *(LoadT*)input_vec;
      }
      else{
        for(int ii = 0, rot_dim_idx = 0; ii < Vec; ii+=2){
          scalar_t x = input_vec[ii];
          scalar_t y = input_vec[ii+1];
          scalar_t cos_val = cos_vec[rot_dim_idx];
          scalar_t sin_val = sin_vec[rot_dim_idx];
          input_vec[ii] = static_cast<scalar_t>(x * cos_val - y * sin_val);
          input_vec[ii+1] = static_cast<scalar_t>(y * cos_val + x * sin_val);
          rot_dim_idx += 1;
        }
        scalar_t* output_ptr = (scalar_t*)kv_cache + kv_cache_offset + 512;
        *(LoadT*)output_ptr = *(LoadT*)input_vec;
        *(LoadT*)(k_pe_ptr) = *(LoadT*)input_vec;
      }
    }
    else
    {
      constexpr int element_per_warp = C10_WARP_SIZE * Vec; // 512
      constexpr int token_per_warp = element_per_warp / (kv_c_dim); // 512 / 512 = 1
      constexpr int token_per_block = token_per_warp * num_warp_per_block; // 1 * 4 = 4
      using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
      using LoadTheta = at::native::memory::aligned_vector<scalar_t, Vec/2>;
      using OutputT = at::native::memory::aligned_vector<uint8_t, Vec>;
      const int warpid = threadIdx.x / C10_WARP_SIZE;
      const int laneid = threadIdx.x % C10_WARP_SIZE;
      const int block_token_id = (blockIdx.x - q_grid_x - k_grid_x) * token_per_block;
      const int warp_token_id = warpid * token_per_warp;
      const int token_id = (blockIdx.x - q_grid_x - k_grid_x) * token_per_block + warp_token_id; // 每个block负责4个token，每个warp负责1个token
      if(token_id >= bs) return;
      // 计算kv_c的offset
      const int thread_offset_in_token = laneid*Vec;
      if(thread_offset_in_token >= kv_c_dim) return;
      // const int kv_c_offset = token_id * k_stride_head + thread_offset_in_token;
      const int kv_c_offset = token_id * k_bs_stride + thread_offset_in_token;
      scalar_t* kv_c_ptr = input_kv_c + kv_c_offset;
      // 计算output的偏移量
      const int64_t slot = slot_mapping[token_id];
      if(slot >= max_slot_id || slot < 0)
      {
        return;
      }
      const int64_t kv_cache_offset = slot * 576 + thread_offset_in_token;
      const int64_t kv_c_normed_offset = token_id * kv_c_dim + thread_offset_in_token;
      scalar_t* kv_c_normed_ptr = output_kv_c_normed + kv_c_normed_offset;
      // 计算weight的offset
      scalar_t* weight_ptr = weight + thread_offset_in_token;
      // 读取kv_c, weight
      scalar_t input_vec[Vec];
      scalar_t weight_vec[Vec];
      uint8_t output_vec[Vec];
      float float_vec[Vec];
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
      if (kv_dtype != 2) // fp8_e5m2 or fp8_e4m3
      {
        #pragma unroll
        for(int ii = 0; ii < Vec; ii+=1){
          float normalized = static_cast<T_ACC>(input_vec[ii]) * s_rstd * static_cast<T_ACC>(weight_vec[ii]);
          if constexpr (APPLY_CLAMP) {
            normalized = clamp_rms_rope_value(normalized, limit);
          }
          float_vec[ii] = normalized;
          input_vec[ii] = static_cast<scalar_t>(normalized);
        }
        if(kv_dtype == 0)
        {
          ((uint32_t*)output_vec)[0] = float_to_bhalf_to_fp8e5m2_x4(float_vec[0], float_vec[1], float_vec[2], float_vec[3]);
          ((uint32_t*)output_vec)[1] = float_to_bhalf_to_fp8e5m2_x4(float_vec[4], float_vec[5], float_vec[6], float_vec[7]);
        }
        else
        {
          ((uint32_t*)output_vec)[0] = float_to_bhalf_to_fp8e4m3_x4(float_vec[0], float_vec[1], float_vec[2], float_vec[3]);
          ((uint32_t*)output_vec)[1] = float_to_bhalf_to_fp8e4m3_x4(float_vec[4], float_vec[5], float_vec[6], float_vec[7]);
        }
        uint8_t* output_ptr = (uint8_t*)kv_cache + kv_cache_offset;
        *(OutputT*)output_ptr = *(OutputT*)output_vec;
        *(LoadT*)kv_c_normed_ptr = *(LoadT*)input_vec;
      }
      else{
        for(int ii = 0; ii < Vec; ii+=1){
          T_ACC normalized = static_cast<T_ACC>(input_vec[ii]) * s_rstd * static_cast<T_ACC>(weight_vec[ii]);
          if constexpr (APPLY_CLAMP) {
            normalized = clamp_rms_rope_value(static_cast<float>(normalized), limit);
          }
          input_vec[ii] = static_cast<scalar_t>(normalized);
        }
        scalar_t* output_ptr = (scalar_t*)kv_cache + kv_cache_offset;
        *(LoadT*)(output_ptr) = *(LoadT*)(input_vec);
        *(LoadT*)kv_c_normed_ptr = *(LoadT*)input_vec;
      }
    }
  }



  void fused_rms_norm_rope_contiguous(torch::Tensor& positions, torch::Tensor& q_pe, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed,
    torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, 
    std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon, double limit) {
    TORCH_CHECK(
        limit == -1.0 || (std::isfinite(limit) && limit >= 0.0),
        "limit must be -1 (disabled) or a finite non-negative value");

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
    const int temp1= 0;
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
    // TORCH_CHECK(k_pe.stride(0) == kv_c_dim + rot_dim, "k_pe.stride(0) must be kv_c_dim + rot_dim");
    const int k_bs_stride = k_pe.stride(0);
    TORCH_CHECK(kv_c_normed.dim() == 2, "kv_c_normed must be 2D tensor [bs, kv_c_dim]");
    TORCH_CHECK(kv_c_normed.size(1) == kv_c_dim, "kv_c_normed.size(1) must be kv_c_dim");
    TORCH_CHECK(kv_c_normed.size(0) == bs, "kv_c_normed.size(0) must be bs");
    TORCH_CHECK(kv_c_normed.stride(0) == kv_c_dim, "kv_c_normed.stride(0) must be kv_c_dim");
    constexpr int element_per_warp = C10_WARP_SIZE * Vec;
    constexpr int k_token_per_warp = element_per_warp / (rot_dim);  // 8
    constexpr int k_token_per_block = k_token_per_warp * num_warp_per_block; // 32
    const int k_grid_x = (bs + k_token_per_block - 1) / k_token_per_block; // (32 + 32 - 1) / 32 = 1
    
    // 计算RMS(K)grid和block尺寸
    TORCH_CHECK(kv_c.dim() == 2, "kv_c must be 2D tensor [bs, kv_c_dim + rot_dim]");
    TORCH_CHECK(kv_c.size(1) == kv_c_dim, "kv_c.size(1) must be kv_c_dim");
    TORCH_CHECK(kv_c.size(0) == bs, "kv_c.size(0) must be bs");
    TORCH_CHECK(kv_c.stride(0) == k_bs_stride, "kv_c.stride(0) must equal k_pe.stride(0)");
    // const int k_bs_stride = kv_c.stride(0);
    constexpr int k_rms_token_per_warp = element_per_warp / (kv_c_dim);// 1
    constexpr int k_rms_token_per_block = k_rms_token_per_warp * num_warp_per_block; // 4
    const int k_rms_grid_x = (bs + k_rms_token_per_block - 1) / k_rms_token_per_block; // (32 + 32 - 1) / 32 = 1
    // 检查KV_CACHE相关参数
    TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D tensor [bs]");
    TORCH_CHECK(slot_mapping.size(0) == bs, "slot_mapping.size(0) must be bs");
    TORCH_CHECK(kv_cache.dim() == 3, "kv_cache must be 3D tensor [num_kv_cache_blocks, kv_cache_blocks_size, kv_c_dim + rot_dim]");
    TORCH_CHECK(kv_cache.size(2) == kv_c_dim + rot_dim, "kv_cache.size(2) must be kv_c_dim + rot_dim");
    const int num_kv_cache_blocks = kv_cache.size(0);
    const int kv_cache_blocks_size = kv_cache.size(1);
    const int max_slot_id = num_kv_cache_blocks * kv_cache_blocks_size;

    TORCH_CHECK(kv_cache_dtype_str == "fp8_e4m3" || kv_cache_dtype_str == "fp8_e5m2" ||
      kv_cache_dtype_str == "fp16" || kv_cache_dtype_str == "bf16" , "kv_cache_dtype_str must be fp8_e4m3 or fp8_e5m2 or fp16 or bf16");
    TORCH_CHECK(kv_cache_scale == 1.0f, "kv_cache_scale must be 1.0");
    int kv_dtype = -1;
    if(kv_cache_dtype_str == "fp8_e5m2"){
      kv_dtype = 0;
    }
    else if(kv_cache_dtype_str == "fp8_e4m3"){
      kv_dtype = 1;
    }
    else {
      kv_dtype = 2;
    }





    const int grid_x = q_grid_x + k_grid_x + k_rms_grid_x;
    // const int grid_x = q_grid_x + k_grid_x;
    dim3 grid(grid_x);
    dim3 block(block_x);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q_pe));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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
        scalar_t* kv_c_normed_ptr = kv_c_normed.data_ptr<scalar_t>();
        void* kv_cache_ptr;
        if (kv_cache.scalar_type() == at::ScalarType::Byte) {
            kv_cache_ptr = kv_cache.data_ptr<uint8_t>();
        } else {
            // 对于 float16/bfloat16，使用相应的指针
            if (kv_cache.scalar_type() == at::ScalarType::Half) {
                kv_cache_ptr = kv_cache.data_ptr<at::Half>();
            } else if (kv_cache.scalar_type() == at::ScalarType::BFloat16) {
                kv_cache_ptr = kv_cache.data_ptr<at::BFloat16>();
            }
        }
        int64_t* slot_mapping_ptr = slot_mapping.data_ptr<int64_t>();
        const scalar_t* cos_sin_cache_ptr = cos_sin_cache.data_ptr<scalar_t>();
        const int64_t* positions_ptr = positions.data_ptr<int64_t>();

        if (limit < 0.0) {
          rms_rope_kernel<scalar_t,T_ACC,Vec,num_warp_per_block,rot_dim,
          kv_c_dim,false><<<grid, block, 0, stream>>>(
              q_pe_ptr,k_pe_ptr,kv_c_ptr,kv_c_normed_ptr,weight_ptr,cos_sin_cache_ptr,
              positions_ptr,slot_mapping_ptr,kv_cache_ptr,
              kv_cache_blocks_size,q_c_dim,num_head,q_stride_head,
              k_stride_head,q_grid_x,k_grid_x, k_rms_grid_x, bs, epsilon,
              static_cast<float>(limit), kv_dtype, max_slot_id, k_bs_stride);
        } else {
          rms_rope_kernel<scalar_t,T_ACC,Vec,num_warp_per_block,rot_dim,
          kv_c_dim,true><<<grid, block, 0, stream>>>(
              q_pe_ptr,k_pe_ptr,kv_c_ptr,kv_c_normed_ptr,weight_ptr,cos_sin_cache_ptr,
              positions_ptr,slot_mapping_ptr,kv_cache_ptr,
              kv_cache_blocks_size,q_c_dim,num_head,q_stride_head,
              k_stride_head,q_grid_x,k_grid_x, k_rms_grid_x, bs, epsilon,
              static_cast<float>(limit), kv_dtype, max_slot_id, k_bs_stride);
        }

      });

}

}
}
