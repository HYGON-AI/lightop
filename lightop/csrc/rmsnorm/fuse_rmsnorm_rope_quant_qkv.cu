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
#include "reduce.h"
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


inline __device__ uint16_t float_to_fp8e4m3_x2(float a, float b)
{
  #if defined(__gfx938__)
    // 对于ROCm，可能需要使用不同的内置函数或组合现有函数
    int32_t temp;
    temp = __builtin_hcu_cvt_pk_fp8_f32(a, b, temp, false);
    return static_cast<uint16_t>(temp & 0xFFFF);
  #else
    uint16_t out;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e4m3(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e4m3(b);
    return out;
  #endif
}


inline __device__ uint16_t float_to_fp8e5m2_x2(float a, float b)
{
  #if defined(__gfx938__)
    // 对于ROCm，可能需要使用不同的内置函数或组合现有函数
    int32_t temp;
    temp = __builtin_hcu_cvt_pk_bf8_f32(a, b, temp, false);
    return static_cast<uint16_t>(temp & 0xFFFF);
  #else
    uint16_t out;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e5m2(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e5m2(b);
    return out;
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

static inline __device__ int8_t float_to_int8_rn(float x) {
  #ifdef USE_ROCM
    static constexpr auto i8_min =
        static_cast<float>(std::numeric_limits<int8_t>::min());
    static constexpr auto i8_max =
        static_cast<float>(std::numeric_limits<int8_t>::max());

    // Use round half away from zero to match PyTorch's behavior for this case
    // For -63.5, we want to round to -63 instead of -64
    float dst = roundf(x);

    // saturate

    // See https://github.com/pytorch/pytorch/issues/127666
    // See https://github.com/llvm/llvm-project/issues/95183
    // hip-clang std::clamp __glibcxx_assert_fail host function when building on
    // Arch/gcc14. The following replaces std::clamp usage with similar logic
    // dst = std::clamp(dst, i8_min, i8_max);
    dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
    return static_cast<int8_t>(dst);
  #else
    // CUDA path
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<const int8_t&>(dst);
  #endif
  }


static inline __device__ float clamp(float x, float min, float max) {
  return fmaxf(fminf(x, max), min);
}


template <typename scalar_t,typename T_ACC, int Vec=8, int num_warp_per_block=4, int rot_dim=64,
int kv_c_dim=512>
__global__ void rmsnorm_rope_quant_qkv_kernel(
    scalar_t* __restrict__ input_q_c,  // [bs, num_head, q_c_dim]
    scalar_t* __restrict__ input_q_pe,  // [bs, num_head, q_c_dim + rot_dim]
    uint8_t* __restrict__ output_q_quant,  // [bs, num_head, q_c_dim + rot_dim]
    float* __restrict__ output_q_scale,  // [bs, num_head]
    scalar_t* __restrict__ input_k,  // [bs, kv_c_dim + rot_dim]
    scalar_t* __restrict__ input_kv_c,  // [bs, kv_c_dim]
    scalar_t* __restrict__ output_kv_c_normed,  // [bs, kv_c_dim]
    scalar_t* __restrict__ weight,  // [kv_c_dim]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim / 2]
    const int64_t* __restrict__ positions,  // [bs]
    const int64_t* __restrict__ slot_mapping,  // [bs]
    uint8_t* __restrict__ kv_cache,  // [num_kv_cache_blocks, kv_cache_blocks_size, kv_c_dim + rot_dim]
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
    const int kv_dtype,
    const int max_slot_id) 
  {
    if(blockIdx.x < q_grid_x){
      constexpr int num_head_per_warp = (C10_WARP_SIZE * Vec) / rot_dim;
      constexpr int num_head_per_block = num_head_per_warp * num_warp_per_block;
      const int block_per_token = (num_head + num_head_per_block - 1) / num_head_per_block;
      using LoadT = at::native::memory::aligned_vector<scalar_t, Vec>;
      using OutputT = at::native::memory::aligned_vector<uint8_t, Vec>;
      using LoadTheta = at::native::memory::aligned_vector<scalar_t, Vec/2>;
      constexpr int embed_dim = rot_dim / 2;
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
      scalar_t* q_pe_ptr = input_q_pe + q_pe_offset;
      uint8_t* q_quant_pe_ptr = output_q_quant + q_pe_offset + q_c_dim;



      // 计算cos_sin地址，并读取cos_sin
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
      const int cos_index_in_cache = element_in_head /2;
      const int sin_index_in_cache = cos_index_in_cache + embed_dim;
      // 读取q_pe、cos、sin
      scalar_t input_vec[Vec];
      scalar_t cos_vec[Vec/2];
      scalar_t sin_vec[Vec/2];
      uint8_t output_vec[Vec];
      float_t rope_vec[Vec];
      if(head_offset_in_token >= num_head) return;
      *(LoadT*)(input_vec) = *(LoadT*)(q_pe_ptr);
      *(LoadTheta*)cos_vec = *(LoadTheta*)(cache_ptr + cos_index_in_cache);
      *(LoadTheta*)sin_vec = *(LoadTheta*)(cache_ptr + sin_index_in_cache);
      #pragma unroll
        for(int ii = 0, rot_dim_idx = 0; ii < Vec; ii+=2){
          scalar_t x = input_vec[ii];
          scalar_t y = input_vec[ii+1];
          scalar_t cos_val = cos_vec[rot_dim_idx];
          scalar_t sin_val = sin_vec[rot_dim_idx];
          rope_vec[ii]   = x * cos_val - y * sin_val;
          rope_vec[ii+1] = y * cos_val + x * sin_val;
          rot_dim_idx += 1;
      }
      if(kv_dtype == 0) // fp8_e5m2
      {
        ((uint32_t*)output_vec)[0] = float_to_fp8e5m2_x4(rope_vec[0], rope_vec[1], rope_vec[2], rope_vec[3]);
        ((uint32_t*)output_vec)[1] = float_to_fp8e5m2_x4(rope_vec[4], rope_vec[5], rope_vec[6], rope_vec[7]);   
      }else{ // fp8_e4m3
        ((uint32_t*)output_vec)[0] = float_to_fp8e4m3_x4(rope_vec[0], rope_vec[1], rope_vec[2], rope_vec[3]);
        ((uint32_t*)output_vec)[1] = float_to_fp8e4m3_x4(rope_vec[4], rope_vec[5], rope_vec[6], rope_vec[7]);      
      }
      // 写回q_pe
      *(OutputT*)q_quant_pe_ptr = *(OutputT*)output_vec;
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
      const int k_pe_offset = token_idx * k_stride_head + thread_offset_in_token;
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
      float_t rope_vec[Vec];
      *(LoadT*)(input_vec) = *(LoadT*)(k_pe_ptr);
      *(LoadTheta*)cos_vec = *(LoadTheta*)(cache_ptr + cos_index_in_cache);
      *(LoadTheta*)sin_vec = *(LoadTheta*)(cache_ptr + sin_index_in_cache);

      #pragma unroll
      for(int ii = 0, rot_dim_idx = 0; ii < Vec; ii+=2){
        scalar_t x = input_vec[ii];
        scalar_t y = input_vec[ii+1];
        scalar_t cos_val = cos_vec[rot_dim_idx];
        scalar_t sin_val = sin_vec[rot_dim_idx];
        // output_vec[ii] = float_to_fp8e5m2(x * cos_val - y * sin_val);
        // output_vec[ii+1] = float_to_fp8e5m2(y * cos_val + x * sin_val);
        rope_vec[ii] = (x * cos_val - y * sin_val);
        rope_vec[ii+1] = (y * cos_val + x * sin_val);
        input_vec[ii] = static_cast<scalar_t>(rope_vec[ii]);
        input_vec[ii+1] = static_cast<scalar_t>(rope_vec[ii+1]);
        rot_dim_idx += 1;
      }

      if (kv_dtype == 0) // fp8_e5m2
      {
        ((uint32_t*)output_vec)[0] = float_to_fp8e5m2_x4(rope_vec[0], rope_vec[1], rope_vec[2], rope_vec[3]);
        ((uint32_t*)output_vec)[1] = float_to_fp8e5m2_x4(rope_vec[4], rope_vec[5], rope_vec[6], rope_vec[7]);
      }
      else // fp8_e4m3
      {
        ((uint32_t*)output_vec)[0] = float_to_fp8e4m3_x4(rope_vec[0], rope_vec[1], rope_vec[2], rope_vec[3]);
        ((uint32_t*)output_vec)[1] = float_to_fp8e4m3_x4(rope_vec[4], rope_vec[5], rope_vec[6], rope_vec[7]);
      }
      uint8_t* output_ptr = (uint8_t*)kv_cache + kv_cache_offset + 512;
      *(OutputT*)output_ptr = *(OutputT*)output_vec;
      *(LoadT*)(k_pe_ptr) = *(LoadT*)input_vec;
    }
    else if(blockIdx.x < q_grid_x+k_grid_x+k_rms_grid_x)
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
      const int kv_c_offset = token_id * k_stride_head + thread_offset_in_token;
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
      float rms_list[Vec];
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
          rms_list[ii] = static_cast<T_ACC>(input_vec[ii]) * s_rstd * static_cast<T_ACC>(weight_vec[ii]);
          input_vec[ii] = static_cast<scalar_t>(rms_list[ii]);
      }
      if (kv_dtype == 0) // fp8_e5m2
      {
        ((uint32_t*)output_vec)[0] = float_to_fp8e5m2_x4(rms_list[0], rms_list[1], rms_list[2], rms_list[3]);
        ((uint32_t*)output_vec)[1] = float_to_fp8e5m2_x4(rms_list[4], rms_list[5], rms_list[6], rms_list[7]);
      }
      else // fp8_e4m3
      {
        ((uint32_t*)output_vec)[0] = float_to_fp8e4m3_x4(rms_list[0], rms_list[1], rms_list[2], rms_list[3]);
        ((uint32_t*)output_vec)[1] = float_to_fp8e4m3_x4(rms_list[4], rms_list[5], rms_list[6], rms_list[7]);
      }
      uint8_t* output_ptr = kv_cache + kv_cache_offset;
      *(OutputT*)output_ptr = *(OutputT*)output_vec;
      *(LoadT*)kv_c_normed_ptr = *(LoadT*)input_vec;
    } 
    else  // 对q_c进行量化
    {
      constexpr int VecQC = 2;
      constexpr int num_head_per_block = 4;
      const int block_per_token = num_head / 4; // 假设num_head = 128, 则block_per_token = 32
      using LoadT = at::native::memory::aligned_vector<scalar_t, VecQC>;
      using OutputT = at::native::memory::aligned_vector<uint8_t, VecQC>;
      const int laneid = threadIdx.x % C10_WARP_SIZE;
      const int warpid = threadIdx.x / C10_WARP_SIZE;
      const int block_idx = (blockIdx.x - q_grid_x - k_grid_x - k_rms_grid_x);
      const int token_idx = block_idx / block_per_token; // 当前线程在第几个token
      if (token_idx >= bs) return;  // 超过batch尺寸直接退出
      const int head_block_idx = block_idx % block_per_token; // 当前block在token中的第几个block
      const int head_start = head_block_idx * num_head_per_block + warpid; // 当前warp在token所负责head编号
      const int thread_offset = laneid * VecQC; // 当前线程在head内处理数据的起始位置
      const int head_offset = head_start * q_stride_head;
      const int all_offset = token_idx * num_head * q_stride_head + head_offset + thread_offset;
      scalar_t* q_c_ptr = input_q_c + all_offset;
      uint8_t* q_quant_c_ptr = output_q_quant + all_offset;
      scalar_t input_vec[VecQC];
      float output_vec_float[VecQC];
      uint8_t output_vec[VecQC];
      if(head_start >= num_head) return;
      *(LoadT*)(input_vec) = *(LoadT*)(q_c_ptr);
      float max_abs_value = 1e-10f;
      max_abs_value = fmaxf(max_abs_value, fabs(static_cast<T_ACC>(input_vec[0])));
      max_abs_value = fmaxf(max_abs_value, fabs(static_cast<T_ACC>(input_vec[1])));
      max_abs_value = WarpReduceMax_ROW<T_ACC,C10_WARP_SIZE>(max_abs_value);
      float fp8_max = kv_dtype == 1 ? 448.0f : 57344.0f;
      float scale_x = max_abs_value / fp8_max;
      scale_x = __shfl(scale_x, 0);
      output_vec_float[0] = clamp(input_vec[0] / scale_x, -fp8_max, fp8_max);
      output_vec_float[1] = clamp(input_vec[1] / scale_x, -fp8_max, fp8_max);
      if(kv_dtype == 0) // fp8_e5m2
      {
        // output_vec[0] = float_to_fp8e5m2(output_vec_float[0]);
        // output_vec[1] = float_to_fp8e5m2(output_vec_float[1]);
        *(uint16_t*)output_vec = float_to_fp8e5m2_x2(output_vec_float[0], output_vec_float[1]);
      }
      else
      {
        // output_vec[0] = float_to_fp8e4m3(output_vec_float[0]);
        // output_vec[1] = float_to_fp8e4m3(output_vec_float[1]);
        *(uint16_t*)output_vec = float_to_fp8e4m3_x2(output_vec_float[0], output_vec_float[1]);
      }
      *(OutputT*)q_quant_c_ptr = *(OutputT*)output_vec;
      if(thread_offset==0)
      {
        output_q_scale[token_idx*num_head + head_start] = scale_x;
      }




    }
  }



  void fuse_rmsnorm_rope_quant_qkv(torch::Tensor& positions, torch::Tensor& q_c, torch::Tensor& q_pe, torch::Tensor& q_quant, torch::Tensor& q_scale, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed,
    torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, 
    std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon) {

    


    const int bs = q_pe.size(0);
    const int num_head = q_pe.size(1);
    constexpr int q_c_dim = 128;
    constexpr int rot_dim = 64;
    const int q_stride_head = q_pe.stride(1);
    TORCH_CHECK(q_c.dim() == 3, "q_c must be 3D tensor [bs, num_head, q_c_dim]");
    TORCH_CHECK(q_pe.dim() == 3, "q_pe must be 3D tensor [bs, num_head, rot_dim]");
    TORCH_CHECK(q_quant.dim() == 3, "q_quant must be 3D tensor [bs, num_head, q_c_dim + rot_dim]");
    TORCH_CHECK(q_stride_head == q_c_dim+rot_dim, "q_stride_head must be q_c_dim + rot_dim");
    TORCH_CHECK(q_pe.size(2) ==  rot_dim, "q_pe.size(2) must be rot_dim");
    TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must be 2D tensor [max_position, rot_dim]");
    TORCH_CHECK(cos_sin_cache.size(1) == rot_dim, "cos_sin_cache.size(1) must equal rot_dim");
    TORCH_CHECK(q_scale.scalar_type() == at::ScalarType::Float, "q_scale must be float tensor");
    const int temp1= 0;
    // 计算ROPE(Q)grid和block尺寸
    // constexpr int num_warp_per_block = 4;
    constexpr int num_warp_per_block = 4;
    constexpr int Vec = 8;
    constexpr int num_head_per_warp = (C10_WARP_SIZE * Vec) / rot_dim;  // 512 / 64 = 8
    constexpr int num_head_per_block = num_head_per_warp * num_warp_per_block;  // 8 * 4 = 32
    const int block_per_token = (num_head + num_head_per_block - 1) / num_head_per_block;  // 向上取整
    const int q_grid_x = block_per_token * bs;  // 4 * bs
    // const int q_c_grid_x = q_grid_x*2;
    const int block_x = num_warp_per_block * C10_WARP_SIZE;  // 128
    // Quant(Q_C)    VecQC = 2, 一个warp处理一个head，一个block处理4个head。 一个token需要 num_head / 4 个block。总共需要 bs * num_head / 4 个block
    const int q_c_grid_x = bs * num_head / 4;
    
    // 计算ROPE(K)grid和block尺寸
    constexpr int kv_c_dim = 512;
    constexpr int k_stride_head = kv_c_dim + rot_dim;
    TORCH_CHECK(k_pe.dim() == 2, "k_pe must be 2D tensor [bs, kv_c_dim + rot_dim]");
    TORCH_CHECK(k_pe.size(1) ==  rot_dim, "k_pe.size(1) must be rot_dim");
    TORCH_CHECK(k_pe.size(0) == bs, "k_pe.size(0) must be bs");
    TORCH_CHECK(k_pe.stride(0) == kv_c_dim + rot_dim, "k_pe.stride(0) must be kv_c_dim + rot_dim");
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
    TORCH_CHECK(kv_c.stride(0) == kv_c_dim + rot_dim, "kv_c.stride(0) must be kv_c_dim + rot_dim");
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
    TORCH_CHECK(kv_cache_dtype_str == "fp8_e4m3" || kv_cache_dtype_str == "fp8_e5m2", "kv_cache_dtype_str must be fp8_e4m3 or fp8_e5m2");
    TORCH_CHECK(kv_cache_scale == 1.0f, "kv_cache_scale must be 1.0");
    int kv_dtype = -1;
    if(kv_cache_dtype_str == "fp8_e5m2"){
      kv_dtype = 0;
    }
    else if(kv_cache_dtype_str == "fp8_e4m3"){    // fp8_e4m3
      kv_dtype = 1;
    }
    else{
      TORCH_CHECK(false, "kv_cache_dtype_str must be fp8_e4m3 or fp8_e5m2");
    }




    const int grid_x = q_grid_x + k_grid_x + k_rms_grid_x + q_c_grid_x;
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
        uint8_t* kv_cache_ptr = kv_cache.data_ptr<uint8_t>();
        uint8_t* q_quant_ptr = q_quant.data_ptr<uint8_t>();
        float* q_scale_ptr = q_scale.data_ptr<float>();
        int64_t* slot_mapping_ptr = slot_mapping.data_ptr<int64_t>();
        const scalar_t* cos_sin_cache_ptr = cos_sin_cache.data_ptr<scalar_t>();
        const int64_t* positions_ptr = positions.data_ptr<int64_t>();
        scalar_t* q_c_ptr = q_c.data_ptr<scalar_t>();
        rmsnorm_rope_quant_qkv_kernel<scalar_t,T_ACC,Vec,num_warp_per_block,rot_dim,
        kv_c_dim><<<grid, block, 0, stream>>>(
            q_c_ptr,q_pe_ptr,q_quant_ptr,q_scale_ptr,k_pe_ptr,kv_c_ptr,kv_c_normed_ptr,weight_ptr,cos_sin_cache_ptr,
            positions_ptr,slot_mapping_ptr,kv_cache_ptr,
            kv_cache_blocks_size,q_c_dim,num_head,q_stride_head,
            k_stride_head,q_grid_x,k_grid_x, k_rms_grid_x, bs, epsilon, kv_dtype, max_slot_id);

      });

}

}
}
