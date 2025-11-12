#pragma once
#include "hip/hip_fp16.h"
#include "hip/hip_bf16.h"
#include "hip/hip_runtime.h"
using bhalf_t = __hip_bfloat16;
using half  = __half;
using BFloat16 = bhalf_t;
using Float16 = half;
using Int32 = int;
using Int16 = unsigned short;
using Float32 = float;
using f8_t = uint8_t;

using half2_t = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
using half4_t = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using half8_t = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using v4bh    = __attribute__((__vector_size__(4 * sizeof(short)))) short;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using uintx4  = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int8x16 = __attribute__((__vector_size__(16 * sizeof(int8_t)))) char;

//fp8_e4m3 definitions
struct alignas(1) Float8_e4m3_t{
    /// Data container
    uint8_t data;
    __host__ __device__ Float8_e4m3_t() = default;
    __host__ __device__ Float8_e4m3_t(uint8_t value): data(value) {}
};

template <typename Element, size_t len>
struct vec {
    using type = __attribute__((__vector_size__(len * sizeof(Element)))) Element;
};

// 特化：为 __half 类型提供专门的实现
template <size_t len>
struct vec<__half, len> {
    using type = __attribute__((__vector_size__(len * sizeof(uint16_t)))) _Float16;
};

// 特化：为 BFloat16 类型提供专门的实现
template <size_t len>
struct vec<BFloat16, len> {
    using type = __attribute__((__vector_size__(len * sizeof(uint16_t)))) unsigned short;
};

// 特化：为 BFloat16 类型提供专门的实现
template <size_t len>
struct vec<int8_t, len> {
    using type = __attribute__((__vector_size__(len * sizeof(int8_t)))) char;
};

// 特化：为 int 类型提供专门的实现
template <size_t len>
struct vec<int, len> {
    using type = __attribute__((__vector_size__(len * sizeof(int)))) int;
};

template <typename Element, size_t len>
union union_vec{
  int8_t int8_array[len * sizeof(Element)];
  Element scalar_array[len];
  typename vec<Element, 2>::type scalar2_array[len/2];
  int int_array[len * sizeof(Element) / 4];
  float float_array[len * sizeof(Element) / 4];
  int32_t uint_array[len * sizeof(Element) / 4];
  int64_t uint64_array[len * sizeof(Element) / 8];
  vec<int8_t, 8>::type int8t_array[len * sizeof(Element) / 8];
  vec<int, 2>::type int2_array[len * sizeof(Element) / 8];
  vec<int, 4>::type int4_array[len * sizeof(Element) / 16];
  vec<float, 4>::type float4_array[len * sizeof(Element) / 16];
};

template<const int kHeadDim, typename T>
__device__ __forceinline__ typename vec<uint, 4>::type tcp_cache_swizzle_func(const T* ptr) {
    typename vec<uint, 4>::type res;
    *(uint64_t*)&res = reinterpret_cast<uint64_t>(ptr); //res[0]放首地址信息
    if constexpr (kHeadDim == 196) {
        res[1] += 0x41800000; // 62 bit: cache swizzle;  48~61: Stride
    } else if constexpr (kHeadDim == 128) {
        res[1] += 0x41000000; // stride 256 Bytes and change tagram
    } else if constexpr (kHeadDim == 64) { // 会走这里
        res[1] += 0x40800000; // stride 128 Bytes and change tagram
    }
    res[2] = 0x80000000;
    res[3] = 0x00020000;
    return res;
}
#define S_BARRIER asm volatile("s_barrier")
////////////////////////////////////////////////////////////////////////////////////////////////////
//封装vmcnt wait
// #define vmcnt_wait(X)\
// __builtin_amdgcn_sched_barrier(0);\
//     asm volatile(\
//       "s_waitcnt vmcnt(%0)\n\t"\
//       "s_barrier\n"\
//       :: "I"(X)\
//       :);\
// __builtin_amdgcn_sched_barrier(0);
// 小i 可以让计数器的范围更大
#define vmcnt_wait(X)\
__builtin_amdgcn_sched_barrier(0);\
    asm volatile(\
      "s_waitcnt vmcnt(%0)\n\t"\
      "s_barrier\n"\
      :: "i"(X)\
      :);\
__builtin_amdgcn_sched_barrier(0);
////////////////////////////////////////////////////////////////////////////////////////////////////
//封装lgkmcnt wait
#define lgkmcnt_wait(X)\
__builtin_amdgcn_sched_barrier(0);\
asm volatile("s_waitcnt lgkmcnt(%0)": : "I"(X));\
__builtin_amdgcn_sched_barrier(0);
////////////////////////////////////////////////////////////////////////////////////////////////////
//封装ds_read2
template<typename VEC, typename dwordx2>
__forceinline__  __device__ void  inline_ds_read2_b32_no_wait(VEC *shared_addr, const int &lds_offset, dwordx2& reg_val, const int offset1) {
  int ldsAddr = reinterpret_cast<size_t>(shared_addr) + lds_offset * 4;
  asm volatile(
      "ds_read2_b32 %0 ,%1 offset0:0 offset1:%2\n"
      : "=v"(reg_val)
      : "v"(ldsAddr), "B"(offset1)
      :);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//封装lgkmcnt wait
#define lgkmcnt_wait_barrier(X)\
__builtin_amdgcn_sched_barrier(0);\
asm volatile(\
  "s_waitcnt lgkmcnt(%0)\n\t"\
  "s_barrier\n"\
  :: "I"(X)\
  :);\
__builtin_amdgcn_sched_barrier(0);

// 共享内存地址转换（32位偏移量）
__device__ __forceinline__ int cvta_to_shared(const void* ptr) {
    return (reinterpret_cast<size_t>(ptr) & 0xFFFFFFFF);
}
// 拷贝4字节（1xfloat）从全局内存到共享内存
__device__ __forceinline__ void cp_async4(float* smem_ptr, uintx4 global_addr, const int gvOffset_s, const int &gvOffset_v) {
    int smem_offset = cvta_to_shared(smem_ptr); //reinterpret_cast<size_t>(smem_ptr);
    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "buffer_load_dword %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(gvOffset_v),
        "s"(smem_offset), "s"(global_addr), "s"(gvOffset_s)
    :);
    // asm volatile(
    //     "s_waitcnt vmcnt(0) \n\t"
    // :);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//封装buffer_load_dword
template<typename src_type=int8_t, typename dst_type=float, const int dword_count=1, const int auxilariy=0>
__forceinline__ __device__ void builtin_buffer_load_dword_lds(src_type *const shared_addr, const typename vec<uint,4>::type rsrc, const int &lds_offset, const int gvOffset_s, const int &gvOffset_v) {
    constexpr int bytes_per_element = sizeof(dst_type);
    dst_type *ptr = reinterpret_cast<dst_type*>(shared_addr) + lds_offset;
    __builtin_amdgcn_raw_buffer_load_lds(
        rsrc,
        (__attribute__((address_space(3))) int*)ptr,
        dword_count * 4, // dword读取
        gvOffset_v * bytes_per_element,
        gvOffset_s * bytes_per_element,
        0, /* immediate offset, instruction offset */
        auxilariy /* auxilariy data| bit 0: glc, bit 1: slc, bit 2: dlc, bit 3: cache swizzle */
    );
    // uintx4 global_ptr = {0};
    // global_ptr[0] = rsrc[0];
    // global_ptr[1] = rsrc[1];
    // global_ptr[2] = rsrc[2];
    // global_ptr[3] = rsrc[3];
    // cp_async4(ptr, global_ptr, gvOffset_s * bytes_per_element, gvOffset_v * bytes_per_element);
}

// buferr_load_reg
template<typename T>	
__forceinline__ __device__ intx4 builtin_amdgcn_buffer_load_reg_dwordx4(const T* ptr, const int vindex, const int offset){ // const int offset
  intx4 rsrc;
  *(uint64_t*)&rsrc = reinterpret_cast<uint64_t>(ptr); //res[0]放首地址信息
  rsrc[1] += 0x40800000;
  rsrc[2] = 0x80000000;
  rsrc[3] = 0x00020000;

  rsrc = __builtin_amdgcn_buffer_load_dwordx4(rsrc, vindex, offset, false, false); // vindx -> sgpr offser->sgpr

//   asm volatile("buffer_load_dwordx4 %0,%1,%2,0, offen offset:0 \n"
//     : "=v"(rsrc), "+v"(offset), "+s"(rsrc));

  return rsrc;
}

// buferr_load_reg
template<typename T>	
__forceinline__ __device__ void buffer_load_reg_dwordx4(const T* ptr, typename vec<int,4>::type & rsrc , const int vindex, int offset){ // const int offset
 
  intx4 global_ptr;
  *(uint64_t*)&global_ptr = reinterpret_cast<uint64_t>(ptr); //res[0]放首地址信息
  global_ptr[1] += 0x40800000;
  global_ptr[2] = 0x80000000;
  global_ptr[3] = 0x00020000;

  asm volatile("buffer_load_dwordx4 %0,%1,%2,0, offen offset:0\n"
    : "=v"(rsrc), "+v"(offset), "+s"(global_ptr));

  return ;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//封装mmac
inline __device__ constexpr int ceil_div(int const& a, int const& b) {
    return (a + b - 1) / b;
}

template<class Element>
__device__ typename vec<float, 4>::type mmac(const typename vec<Element, 4>::type &v1, const typename vec<Element, 4>::type &v2, const typename vec<float, 4>::type &v3)
{
    floatx4 v4f;
    #if defined(__gfx936__) || defined(__gfx928__)
        v4f = __builtin_amdgcn_mmac_f32_16x16x16f16(v1, v2, v3);
    #endif
    return v4f;
}

template<class Element>
__device__ typename vec<int, 4>::type mmac(const typename vec<Element, 8>::type &v1, const typename vec<Element, 8>::type &v2, const typename vec<int, 4>::type &v3)
{
    auto a = reinterpret_cast<const long*>(&v1);
    auto b = reinterpret_cast<const long*>(&v2);
    intx4 v4i;
    // __builtin_amdgcn_sched_barrier(0);
    #if defined(__gfx936__) || defined(__gfx928__)
        v4i = __builtin_amdgcn_mmac_i32_16x16x32i8(*a, *b, v3);
    #endif
    // __builtin_amdgcn_sched_barrier(0);
    return v4i;
}

template<class Element>
__device__ typename vec<int, 4>::type mmac_int8(const typename vec<Element, 8>::type &v1, const typename vec<Element, 8>::type &v2, typename vec<int, 4>::type &v3)
{
    
    auto a = reinterpret_cast<const long*>(&v1);
    auto b = reinterpret_cast<const long*>(&v2);
    #if defined(__gfx936__) || defined(__gfx928__)
    __builtin_amdgcn_sched_barrier(0);
    asm volatile(
        "v_mmac_i32_16x16x32_i8 %0, %1, %2, %0 \n"
        : "+v"(v3)
        : "v"(*a), "v"(*b)
      );
    __builtin_amdgcn_sched_barrier(0);
    #endif
    return v3;
}

template<>
__device__ typename vec<float, 4>::type mmac<half>(const typename vec<half, 4>::type &v1, const typename vec<half, 4>::type &v2, const typename vec<float, 4>::type &v3)
{
    floatx4 v4f;
    #if defined(__gfx936__) || defined(__gfx928__)
        v4f = __builtin_amdgcn_mmac_f32_16x16x16f16(v1, v2, v3);
    #endif
    return v4f;

}

template<>
__device__ typename vec<int, 4>::type mmac<int8_t>(const typename vec<int8_t, 8>::type &v1, const typename vec<int8_t, 8>::type &v2, const typename vec<int, 4>::type &v3)
{
    auto a = reinterpret_cast<const long*>(&v1);
    auto b = reinterpret_cast<const long*>(&v2);
    intx4 v4i;
    #if defined(__gfx936__) || defined(__gfx928__)
        v4i = __builtin_amdgcn_mmac_i32_16x16x32i8(*a, *b, v3);
    #endif
    return v4i;
}

template<>
__device__ typename vec<float, 4>::type mmac<bhalf_t>(const typename vec<bhalf_t, 4>::type &v1, const typename vec<bhalf_t, 4>::type &v2, const typename vec<float, 4>::type &v3)
{
    floatx4 v4f;
    #if defined(__gfx936__) || defined(__gfx928__)
        v4f = __builtin_amdgcn_mmac_f32_16x16x16bf16(v1, v2, v3);
    #endif
    return v4f;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//float和e4m3互转
#if !defined(__NVCC__)
// fp8_e5m2
constexpr int32_t e5m2_exp_bits  = 5;
constexpr int32_t e5m2_mant_bits = 2;
constexpr int32_t e5m2_bits      = 8;
constexpr int32_t e5m2_bias      = (1 << (e5m2_exp_bits - 1)) - 1;
// fp8_e4m3
constexpr int32_t e4m3_exp_bits  = 4;
constexpr int32_t e4m3_mant_bits = 3;
constexpr int32_t e4m3_bits      = 8;
constexpr int32_t e4m3_bias      = (1 << (e4m3_exp_bits - 1)) - 1;
// fp16
constexpr int32_t fp16_exp_bits  = 5;
constexpr int32_t fp16_mant_bits = 10;
constexpr int32_t fp16_bits      = 16;
constexpr int32_t fp16_bias      = (1 << (fp16_exp_bits - 1)) - 1;
// fp32
constexpr int32_t fp32_exp_bits  = 8;
constexpr int32_t fp32_mant_bits = 23;
constexpr int32_t fp32_bits      = 32;
constexpr int32_t fp32_bias      = (1 << (fp32_exp_bits - 1)) - 1;

__host__ __device__
static uint8_t __float2e4m3(const float src) {
    // conversion from float to unsigned int(32 bits) for convience to fetching each bit
    uint32_t __src = *(unsigned int*)&src;
    // fetch sign bits
    uint8_t sign_bits = (__src & 0x80000000) >> (fp32_bits - e5m2_bits);
    // fetch exponent bitss
    uint32_t exp_bits  = __src & 0x7f800000;
    // fetch mantissa bits
    uint32_t mant_bits = __src & 0x007fffff;
    // fetch absolute value
    uint32_t data_scale = __src & 0x7fffffff;
    // categorical discussions
    /* NAN */
    uint8_t result = 0x0;
    if (exp_bits == 0x7f800000 and mant_bits > 0x0) {
        // result = sign_bits | 0x7f; // output qNAN
        result = 0x7f; // for NV's __nv_cvt_float_to_fp8:cvt.rn.satfinite.e4m3x2.f32, output are all 0x7f
    }
    /* inf or greater than MAX value of E5M2 */
    else if ((exp_bits == 0x7f800000 and mant_bits == 0x0) or (data_scale > 0x43e00000)) {
        result = sign_bits | 0x7e; // output MAX
    }
    /* less than MIN of denorm */
    else if (exp_bits <= 0x3a800000) {
        result = (exp_bits == 0x3a800000 and mant_bits > 0x0) ? sign_bits | 0x1: sign_bits;
    }
    /* others */
    else {
        /* norm fp32 can be represented as denorm fp8 / norm */
        mant_bits = exp_bits < 0x3c800000 ? (0x800000 | mant_bits) >> ((0x3c800000 - exp_bits) >> fp32_mant_bits): mant_bits;
        exp_bits  = exp_bits < 0x3c800000 ? 0x0: ((exp_bits >> fp32_mant_bits) - (fp32_bias - e4m3_bias)) << e4m3_mant_bits;
        // get discard bits
        uint32_t discard_bits = mant_bits & 0xfffff;
        // rounding
        bool carry_a_bit = discard_bits > 0x80000 or (discard_bits == 0x80000 and (mant_bits & 0x100000));
        mant_bits = (mant_bits & 0x700000) >> (fp32_mant_bits - e4m3_mant_bits);
        mant_bits = carry_a_bit ? mant_bits + 1: mant_bits;
        result = sign_bits + exp_bits + mant_bits; // + rather than |, since mant may carry a bit to exp
    }
    return result;
}

__host__ __device__
static float __e4m32float(const uint8_t src) {
    // initialize ret value
    float result;
    // conversion from float to unsigned int(32 bits) for convience to fetching each bit
    uint8_t __src = *(uint8_t*)&src;
    // fetch sign bits
    uint32_t sign_bits = __src & 0x80;
    // fetch exponent bits
    uint32_t exp_bits  = (__src & 0x78) >> e4m3_mant_bits;
    // fetch mantissa bits
    uint32_t mant_bits = __src & 0x7;
    // denorm or 0
    if (exp_bits == 0x0 and mant_bits >= 0x0) {
        result = 0.0078125f * ((mant_bits & 0x4) >> 2) + 0.00390625f * ((mant_bits & 0x2) >> 1) + 0.001953125f * (mant_bits & 0x1);
        result = sign_bits > 0 ? -result: result;
    } else {
        uint32_t tmp = (exp_bits == 0xf and mant_bits == 0x7)
            ? /*input NaN*/0x7fffffff
            : /*input norm*/(sign_bits << (fp32_bits - e4m3_bits)) + ((exp_bits - e4m3_bias + fp32_bias) << fp32_mant_bits) + (mant_bits << (fp32_mant_bits - e4m3_mant_bits));
        result = *(float*)&tmp;
    }
    return result;
}
#endif // end of #if !defined(__NVCC__)

////////////////////////////////////////////////////////////////////////////////////////////////////
static __device__ inline uint16_t f32_to_bf16(float f){
  
    uint32_t u = *(uint32_t*)(&f);
    u += 0x7fff + ((u >> 16) & 1); 
    return u>>16;

}

//f32转bf16
// to simplify f32 -> bf16 conversion, filter some branch
// inline __HOST_DEVICE__ bhalf_t inlineasm_float2bfloat16_nonan(const float f) {
// 	bhalf_t ret;
// #if defined(__gfx938__)
//     ret.data = __builtin_hygon_cvt_bf16_f32(f, /*clamp*/false, /*dst_sel*/false);
// // #elif __HIP_DEVICE_COMPILE__
// // inline asm may lead to spill in scratch memory
// #elif 0
//     unsigned int help = 0x7fff; // this line can be optimized, so as to use v_add3_u32
//     unsigned int tmp;
//     asm volatile(
//         "v_lshrrev_b32 %0, 16, %1\n\t"
//         "v_and_b32 %0, 0x1, %0\n\t"
//         : "=v"(tmp)
//         : "v"(f)
//         :);
//     asm volatile(
//         "v_add3_u32 %0, %2, %3, %4\n"
//         "v_lshrrev_b32 %1, 16, %0\n"
//         : "=v"(tmp), "=v"(ret.data)
//         : "v"(tmp), "s"(help), "v"(f)
//         :);
// #else
//     // for inf, 0x7f80-0000 + 0x0000-7fff + (0x7f80 & 1) = 0x7f80-7ffff, and >> 16 -> 0x7f80, still inf
//     // for nan, no process, for input is from softmax, > 0 and no nan
//     // for others, actually, not totally rounding to nearest even, no case of mantissa 1000
//     union {
// 		float fp32;
// 		unsigned int u32;
// 	} u = {f};
// 	u.u32 += 0x7fff + ((u.u32 >> 16) & 1);
// 	ret.data = (u.u32 >> 16);
// #endif
// 	return ret;
// }
////////////////////////////////////////////////////////////////////////////////////////////////////
//数据类型转化封装
//DownCast
//fp32转fp16
template<class FromType, class ToType, bool Is_short = false,typename std::enable_if<std::is_same<FromType,float>::value && std::is_same<ToType,half>::value, int>::type = 0>
__host__ __device__ ToType DownCast(const FromType &from_var) {
    return __float2half(from_var);
}
//fp32转bf16，并返回其内置数据类型
// template<class FromType, class ToType, bool Is_short = false,typename std::enable_if<std::is_same<FromType,float>::value && Is_short && std::is_same<ToType, BFloat16>::value, int>::type = 0>
// __host__ __device__ unsigned short DownCast(const FromType &from_var) {
//     return inlineasm_float2bfloat16_ushort_nonan(from_var);
//     // return __float2bfloat16(from_var).data;
// }
// //fp32转bf16，返回其结构体本身
// template<class FromType, class ToType, bool Is_short = false,typename std::enable_if<std::is_same<FromType,float>::value && !Is_short && std::is_same<ToType, BFloat16>::value, int>::type = 0>
// __host__ __device__ BFloat16 DownCast(const float &from_var) {
//     return inlineasm_float2bfloat16_nonan(from_var);
//     // return __float2bfloat16(from_var);
// }
//fp32转fp8，返回其内置数据类型
template<class FromType, class ToType, bool Is_uint8 = false,typename std::enable_if<std::is_same<FromType,float>::value && Is_uint8 && std::is_same<ToType, Float8_e4m3_t>::value, int>::type = 0>
__host__ __device__ uint8_t DownCast(const float &from_var) {
    return __float2e4m3(from_var);
}
//fp32转fp8，返回其结构体本身
template<class FromType, class ToType, bool Is_uint8 = false,typename std::enable_if<std::is_same<FromType,float>::value && !Is_uint8 && std::is_same<ToType, Float8_e4m3_t>::value, int>::type = 0>
__host__ __device__ Float8_e4m3_t DownCast(const float &from_var) {
    return Float8_e4m3_t(__float2e4m3(from_var));
}
//fp16转fp8，返回其内置数据类型
template<class FromType, class ToType, bool Is_uint8 = false,typename std::enable_if<std::is_same<FromType,half>::value && Is_uint8 && std::is_same<ToType, Float8_e4m3_t>::value, int>::type = 0>
__host__ __device__ uint8_t DownCast(const half &from_var) {
    float src_f32 = __half2float(from_var);
    return __float2e4m3(src_f32);
}
//fp16转fp8，返回其结构体本身
template<class FromType, class ToType, bool Is_uint8 = false,typename std::enable_if<std::is_same<FromType,half>::value && !Is_uint8 && std::is_same<ToType, Float8_e4m3_t>::value, int>::type = 0>
__host__ __device__ Float8_e4m3_t DownCast(const half &from_var) {
    float src_f32 = __half2float(from_var);
    return Float8_e4m3_t(__float2e4m3(src_f32));
}


//UpCast
//fp16转fp32
template<class FromType=half, class ToType=float, bool Is_short = false,typename std::enable_if<std::is_same<FromType,half>::value && std::is_same<ToType,float>::value, int>::type = 0>
__host__ __device__ float UpCast(const half &from_var) {
    return __half2float(from_var);
}
// //bf16的内置数据类型转fp32
// template<class FromType, class ToType, bool Is_short = false,typename std::enable_if<Is_short && std::is_same<FromType, BFloat16>::value && std::is_same<ToType,float>::value, int>::type = 0>
// __host__ __device__ float UpCast(const unsigned short &from_var) {
//     struct __hip_bfloat16 x;
//     x.data = from_var;
//     return  __bfloat162float(x);
// }
//bf16转fp32
template<class FromType, class ToType, bool Is_short = false,typename std::enable_if<!Is_short && std::is_same<FromType, BFloat16>::value && std::is_same<ToType,float>::value, int>::type = 0>
__host__ __device__ float UpCast(const BFloat16 &from_var) {
    return __bfloat162float(from_var);
}
//fp8的内置数据类型转fp32
template<class FromType, class ToType, bool Is_uint8 = false,typename std::enable_if<Is_uint8 && std::is_same<FromType, Float8_e4m3_t>::value && std::is_same<ToType,float>::value, int>::type = 0>
__host__ __device__ float UpCast(const uint8_t &from_var) {
    return __e4m32float(from_var);
}
//fp8转fp32
template<class FromType, class ToType, bool Is_uint8 = false,typename std::enable_if<!Is_uint8 && std::is_same<FromType, Float8_e4m3_t>::value && std::is_same<ToType,float>::value, int>::type = 0>
__host__ __device__ float UpCast(const Float8_e4m3_t &from_var) {
    return __e4m32float(from_var.data);
}
//fp8的内置数据类型转fp16
template<class FromType, class ToType, bool Is_uint8 = false,typename std::enable_if<Is_uint8 && std::is_same<FromType, Float8_e4m3_t>::value && std::is_same<ToType,half>::value, int>::type = 0>
__host__ __device__ half UpCast(const uint8_t &from_var) {
    float src_f32 = __e4m32float(from_var);
    return __float2half(src_f32);
}
//fp8转fp16
template<class FromType, class ToType, bool Is_uint8 = false,typename std::enable_if<!Is_uint8 && std::is_same<FromType, Float8_e4m3_t>::value && std::is_same<ToType,half>::value, int>::type = 0>
__host__ __device__ half UpCast(const Float8_e4m3_t &from_var) {
    float src_f32 = __e4m32float(from_var.data);
    return __float2half(src_f32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//封装divide函数
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

///////////////////////////////////////////////block swiizzle/////////////////////////////////////////////////////
__device__ __forceinline__ void GetBLockIdx(
      int32_t loop_idx, int32_t m_loop, int32_t n_loop, int32_t swizzl_direction, int32_t swizzl_count,
      int &m_idx, int &n_idx
  ) {
    int32_t in_batch_idx = loop_idx % (m_loop * n_loop);
    if (swizzl_direction == 0) { // Zn
        int32_t tile_block_loop = (m_loop + swizzl_count -1) / swizzl_count;
        int32_t tile_block_idx = in_batch_idx / (swizzl_count * n_loop);
        int32_t in_tile_block_idx = in_batch_idx % (swizzl_count * n_loop);
        int32_t n_row = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1) {
            n_row = m_loop - swizzl_count * tile_block_idx;
        }
        m_idx = tile_block_idx * swizzl_count + in_tile_block_idx % n_row;
        n_idx = in_tile_block_idx / n_row;
    } else if (swizzl_direction == 1) { //Nz
        int32_t tile_block_loop = (n_loop+swizzl_count -1) / swizzl_count;
        int32_t tile_block_idx = in_batch_idx / (swizzl_count * m_loop);
        int32_t in_tile_block_idx = in_batch_idx % (swizzl_count * m_loop);
        int32_t n_col = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1) {
            n_col = n_loop - swizzl_count * tile_block_idx;
        }
        n_idx = tile_block_idx * swizzl_count + (in_tile_block_idx % n_col);
        m_idx = (in_tile_block_idx / n_col);
    } else if (swizzl_direction == 2) { // Zz
        int32_t tile_block_loop = (m_loop + swizzl_count -1) / swizzl_count;
        int32_t tile_block_idx = in_batch_idx / (swizzl_count * n_loop);
        int32_t in_tile_block_idx = in_batch_idx % (swizzl_count * n_loop);
        int32_t n_row = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1) {
            n_row = m_loop - swizzl_count * tile_block_idx;
        }
        int32_t n_tile_block_loop = (n_loop + swizzl_count -1) / swizzl_count;
        int32_t n_tile_block_idx = in_tile_block_idx / (n_row * swizzl_count);
        int32_t n_in_tile_block_idx = in_tile_block_idx % (n_row * swizzl_count);
        int32_t n_col = swizzl_count;
        if (n_tile_block_idx == n_tile_block_loop - 1) {
            n_col = n_loop - swizzl_count * n_tile_block_idx;
        }
        m_idx = tile_block_idx * swizzl_count + n_in_tile_block_idx / n_col;
        n_idx = n_tile_block_idx * swizzl_count + (n_in_tile_block_idx % n_col);
    } else if (swizzl_direction == 3) { // Nn
        int32_t tile_block_loop = (n_loop+swizzl_count -1) / swizzl_count;
        int32_t tile_block_idx = in_batch_idx / (swizzl_count * m_loop);
        int32_t in_tile_block_idx = in_batch_idx % (swizzl_count * m_loop);
        int32_t n_col = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1) {
            n_col = n_loop - swizzl_count * tile_block_idx;
        }
        int32_t m_tile_block_loop = (m_loop + swizzl_count -1) / swizzl_count;
        int32_t m_tile_block_idx = in_tile_block_idx / (n_col * swizzl_count);
        int32_t m_in_tile_block_idx = in_tile_block_idx % (n_col * swizzl_count);
        int32_t n_row = swizzl_count;
        if (m_tile_block_idx == m_tile_block_loop - 1) {
            n_row = m_loop - swizzl_count * m_tile_block_idx;
        }
        m_idx = m_tile_block_idx * swizzl_count + m_in_tile_block_idx % n_row;
        n_idx = tile_block_idx * swizzl_count + m_in_tile_block_idx / n_row;
    } else{
      int blocks_per_tile = n_loop * swizzl_count;
      int num_tiles = m_loop / swizzl_count;
      int block_idx_flatterned = n_idx * m_loop + m_idx;
      int tile_id = block_idx_flatterned / blocks_per_tile;
      int block_idx_in_tile = block_idx_flatterned % blocks_per_tile;
      int block_idx_x_in_tile = block_idx_in_tile % swizzl_count;
      int block_idx_y_in_tile = block_idx_in_tile / swizzl_count;
      if (m_idx >= num_tiles * swizzl_count) {
        int last_tile_dim_x = m_loop - num_tiles * swizzl_count;
        block_idx_x_in_tile = block_idx_in_tile % last_tile_dim_x;
        block_idx_y_in_tile = block_idx_in_tile / last_tile_dim_x;
      }
      
      int swizzled_block_idx_flatterned =
          block_idx_y_in_tile * m_loop + block_idx_x_in_tile + tile_id * swizzl_count;

      m_idx = swizzled_block_idx_flatterned % m_loop;
      n_idx = swizzled_block_idx_flatterned / m_loop;
    
    }
  }
