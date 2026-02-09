#pragma once

#include "reduce.h"
#include "hip/hip_bf16.h"
#include <sys/syscall.h>
#include <unistd.h>


#define Output_Type_SWITCH(DST_DTYPE, ...)     \
  [&] {                                       \
    using scalar_t = _Float16;                \
    return __VA_ARGS__();                     \
  }()

#define KPerWarp_SWITCH(k_per_warp, ...)    \
  [&] {                                       \
    if (k_per_warp == 256) {                  \
      constexpr static int KPerWarp = 256;      \
      return __VA_ARGS__();                     \
    } else {                  \
      constexpr static int KPerWarp = 128;      \
      return __VA_ARGS__();                     \
    }                                            \
  }()

  #define NUM_N_WARPS_SWITCH(num_n_warps, ...)          \
  [&] {                                             \
    if (num_n_warps == 1) {                         \
      constexpr static int NUM_N_WARPS = 1;      \
      return __VA_ARGS__();                   \
    } else{                  \
      constexpr static int NUM_N_WARPS = 2;      \
      return __VA_ARGS__();                   \
    }                                                 \
  }()


#define NUM_K_WARPS_SWITCH(num_k_warps, ...)          \
  [&] {                                             \
    if (num_k_warps == 1) {                         \
      constexpr static int NUM_K_WARPS = 1;      \
      return __VA_ARGS__();                   \
    } else if (num_k_warps == 2) {                  \
      constexpr static int NUM_K_WARPS = 2;      \
      return __VA_ARGS__();                   \
    } else if (num_k_warps == 4) {                  \
      constexpr static int NUM_K_WARPS = 4;      \
      return __VA_ARGS__();                   \
    } else if(num_k_warps == 8) {                  \
      constexpr static int NUM_K_WARPS = 8;      \
      return __VA_ARGS__();                   \
    } else {                                   \
      constexpr static int NUM_K_WARPS = 16;      \
      return __VA_ARGS__();                   \
    }                                           \
  }()

#define NPerWarp_SWITCH(n_per_warp, ...)    \
[&] {                                       \
  if (n_per_warp == 16) {                  \
    constexpr static int NPerWarp = 16;    \
    return __VA_ARGS__();                   \
  } else if (n_per_warp == 32) {                  \
    constexpr static int NPerWarp = 32;    \
    return __VA_ARGS__();                   \
  } else if(n_per_warp == 64){                                   \
    constexpr static int NPerWarp = 64;    \
    return __VA_ARGS__();                   \
  } else {                                   \
    constexpr static int NPerWarp = 128;    \
    return __VA_ARGS__();                   \
  }                                           \
}()
#define FastCompile 0
#if FastCompile

#define UseMarlin_SWITCH(use_marlin, ...)          \
[&] {                                       \
  constexpr static bool UseMarlin = true;        \
  return __VA_ARGS__();                     \
}()



#else
#define UseMarlin_SWITCH(use_marlin, ...)          \
[&] {                                       \
  if (use_marlin) {                          \
    constexpr static bool UseMarlin = true;        \
    return __VA_ARGS__();                     \
  } else {                                   \
    constexpr static bool UseMarlin = false;        \
    return __VA_ARGS__();                     \
  }                                           \
}()
#endif



using float4_t = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
using half4_t = __attribute__( (__vector_size__(4 * sizeof(_Float16)) )) _Float16;
using bhalf4_t = __attribute__( (__vector_size__(4 * sizeof(short)) )) short;
using half2_t = __attribute__( (__vector_size__(2 * sizeof(_Float16)) )) _Float16;
using half8_t = __attribute__( (__vector_size__(8 * sizeof(_Float16)) )) _Float16;
using half_t = _Float16;

using int32x4_t = __attribute__( (__vector_size__(4 * sizeof(int32_t)) )) int32_t;

inline __device__ void builtin_amdgcn_mmac(const half4_t& reg_a, const half4_t& reg_b, float4_t& reg_c)
{
  #if defined(__gfx936__) || defined(__gfx928__)
  reg_c = __builtin_amdgcn_mmac_f32_16x16x16f16(reg_a,reg_b,reg_c);
  #elif defined(__gfx938__)
  reg_c = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(reg_a,reg_b,reg_c, false, false);
  #endif
}


inline __device__ void builtin_amdgcn_mmac(const bhalf4_t& reg_a, const bhalf4_t& reg_b, float4_t& reg_c)
{
  #if defined(__gfx936__) || defined(__gfx928__)
  reg_c = __builtin_amdgcn_mmac_f32_16x16x16bf16(reg_a,reg_b,reg_c);
  #elif defined(__gfx938__)
  reg_c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(reg_a,reg_b,reg_c, false, false);
  #endif
}

template<typename VecType, typename VecQuantType, typename VecScaleZeroType, typename scalar_t, int tc_number>
inline __device__ void dequant_func(VecType& b_vec, VecQuantType& b_quant_vec, VecScaleZeroType& b_scale_zero_vec, int G)
{
  static constexpr uint32_t BOT_MASK    = 0x000f000f;
  static constexpr uint32_t TOP_MASK    = 0x00f000f0;
  static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;
  static constexpr uint32_t MAGIC_NUM_2 = MAGIC_NUM_1 >> 4;
  // 这里假设G=32来写的。G=64或者128的话，可以通过重排scale_zero来实现
  for(int i=0;i<tc_number/2;i++){
    const int32_t source = b_quant_vec.data[i];
    const int32_t top_i4s = source >> 8;
    b_vec.data_int32[i][0] = ((source & BOT_MASK) | MAGIC_NUM_2)<<4;
    b_vec.data_int32[i][1] = (source & TOP_MASK) | MAGIC_NUM_1;
    b_vec.data_int32[i][2] = ((top_i4s & BOT_MASK) | MAGIC_NUM_2)<<4;
    b_vec.data_int32[i][3] = (top_i4s & TOP_MASK) | MAGIC_NUM_1;
    b_vec.data_scalar[2*i][0] = (b_vec.data_scalar[2*i][0] - b_scale_zero_vec.data_scalar[i][0]) * b_scale_zero_vec.data_scalar[i][1];
    b_vec.data_scalar[2*i+1][0] = (b_vec.data_scalar[2*i+1][0] - b_scale_zero_vec.data_scalar[i][0]) * b_scale_zero_vec.data_scalar[i][1];
    b_vec.data_scalar[2*i][1] = (b_vec.data_scalar[2*i][1] - b_scale_zero_vec.data_scalar[i][0]) * b_scale_zero_vec.data_scalar[i][1];
    b_vec.data_scalar[2*i+1][1] = (b_vec.data_scalar[2*i+1][1] - b_scale_zero_vec.data_scalar[i][0]) * b_scale_zero_vec.data_scalar[i][1];
    b_vec.data_scalar[2*i][2] = (b_vec.data_scalar[2*i][2] - b_scale_zero_vec.data_scalar[i][0]) * b_scale_zero_vec.data_scalar[i][1];
    b_vec.data_scalar[2*i+1][2] = (b_vec.data_scalar[2*i+1][2] - b_scale_zero_vec.data_scalar[i][0]) * b_scale_zero_vec.data_scalar[i][1];
    b_vec.data_scalar[2*i][3] = (b_vec.data_scalar[2*i][3] - b_scale_zero_vec.data_scalar[i][0]) * b_scale_zero_vec.data_scalar[i][1];
    b_vec.data_scalar[2*i+1][3] = (b_vec.data_scalar[2*i+1][3] - b_scale_zero_vec.data_scalar[i][0]) * b_scale_zero_vec.data_scalar[i][1];
  }
}



inline __device__ void dequantize_s4_to_fp16x2(uint32_t const& source, uint32_t* h) {
  static constexpr uint32_t BOT_MASK    = 0x000f000f;
  static constexpr uint32_t TOP_MASK    = 0x00f000f0;
  static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;       // `64`
  static constexpr uint32_t MAGIC_NUM_2 = MAGIC_NUM_1 >> 4; // `64` >> 4
  const uint32_t top_i4s                = source >> 8;

  h[0] = ((source & BOT_MASK) | MAGIC_NUM_2)<<4;
  h[1] = (source & TOP_MASK) | MAGIC_NUM_1;
  h[2] = ((top_i4s & BOT_MASK) | MAGIC_NUM_2)<<4;
  h[3] = (top_i4s & TOP_MASK) | MAGIC_NUM_1;
}


#define MIN(a,b) ((a) < (b) ? (a) : (b))