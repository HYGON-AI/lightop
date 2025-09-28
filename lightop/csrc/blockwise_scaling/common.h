/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#if defined(__gfx936__) || defined(__gfx928__)
  #include <hip/hip_fp8.h>
#endif

#include <cuda_runtime_api.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace at {
namespace native {

enum class DType {
  kByte = 0,
  kInt16 = 1,
  kInt32 = 2,
  kInt64 = 3,
  kFloat32 = 4,
  kFloat16 = 5,
  kBFloat16 = 6,
  kFloat8E4M3 = 7,
  kFloat8E5M2 = 8,
  kFloat8E8M0 = 9,
  kFloat4E2M1 = 10,
  kInt8 = 11,
  kNumTypes
};

using byte = uint8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using fp32 = float;
using fp16 = half;
using int8 = int8_t;
using bf16 = __hip_bfloat16;

#if defined(__gfx936__) || defined(__gfx928__)
using fp8e4m3 = __hip_fp8_e4m3;
using fp8e5m2 = __hip_fp8_e5m2;
#else
using fp8e4m3 = uint8_t;
using fp8e5m2 = uint8_t;
#endif


template <typename T>
constexpr T DIVUP(const T &x, const T &y) {
  return (((x) + ((y)-1)) / (y));
}

// enum class for rowwise usage
enum class FP8BlockwiseRowwiseOption {
  // No rowwise data, skip rowwise quantization
  NONE,
  // Rowwise data, scales in GEMM format
  ROWWISE_GEMM_READY,
  // Rowwise data, scales in compact format, needs extra processing (padding, transposing) before GEMM
  ROWWISE_COMPACT
};

// enum class for columnwise usage
// For Hopper sm90 with only TN fp8 gemm, there is need to do columnwise transpose when doing 1D block scaling
enum class FP8BlockwiseColumnwiseOption {
  // No columnwise data, skip columnwise quantization
  NONE,
  // Columnwise data transposed from original shape.
  // Scales in GEMM format corresponding to GEMM ingesting transposed column data.
  // On Hopper sm90, GEMM_READY means that columnwise quantization also fuses transpose op
  // On higher sm versions with TN,NT,NN fp8 gemm, GEMM_READY doesn't fuse transpose
  COLUMNWISE_GEMM_READY,
  // Columnwise data in original shape
  // Scales in compact format, needs extra processing (padding, transposing) before GEMM
  COLUMNWISE_COMPACT
};


template <typename T>
constexpr inline const char *type_name() noexcept;
#define TRANSFORMER_ENGINE_TYPE_NAME(T)                  \
  template <>                                            \
  inline constexpr const char *type_name<T>() noexcept { \
    return #T;                                           \
  }
TRANSFORMER_ENGINE_TYPE_NAME(uint8_t)
TRANSFORMER_ENGINE_TYPE_NAME(int16_t)
TRANSFORMER_ENGINE_TYPE_NAME(int32_t)
TRANSFORMER_ENGINE_TYPE_NAME(int64_t)
TRANSFORMER_ENGINE_TYPE_NAME(float)
TRANSFORMER_ENGINE_TYPE_NAME(__half)
TRANSFORMER_ENGINE_TYPE_NAME(__hip_bfloat16)
#if defined(__gfx936__) || defined(__gfx928__)
TRANSFORMER_ENGINE_TYPE_NAME(__hip_fp8_e4m3)
TRANSFORMER_ENGINE_TYPE_NAME(__hip_fp8_e5m2)
#endif
TRANSFORMER_ENGINE_TYPE_NAME(int8_t)
#undef TRANSFORMER_ENGINE_TYPE_NAME


template <typename T>
struct TypeExtrema;

template <>
struct TypeExtrema<int8> {
  static constexpr float max = 127.0f;
};

#if defined(__gfx936__) || defined(__gfx928__)
template <>
struct TypeExtrema<fp8e4m3> {
  static constexpr float max = 448.0f;
};

template <>
struct TypeExtrema<fp8e5m2> {
  static constexpr float max = 57344.0f;
};
#endif

template <>
struct TypeExtrema<bf16> {
  // Hex float format of 1.(7 bits of 1) * 2 ^ 127
  static constexpr float max = 0x1.FEp127;
};

template <>
struct TypeExtrema<fp16> {
  // Hex float format of 1.(10 bits of 1) * 2 ^ 15
  static constexpr float max = 0x1.FFCp15;
};

template <typename T>
struct TypeExtrema {
  static constexpr float max = std::numeric_limits<T>::max();
};


template <typename T>
struct BitsNumber;

template <typename T>
struct BitsNumber {
  static constexpr size_t num_bits = 8 * sizeof(T);
};



inline const std::string &to_string_like(const std::string &val) noexcept { return val; }

constexpr const char *to_string_like(const char *val) noexcept { return val; }

/* \brief Convert arithmetic type to string */
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline std::string to_string_like(const T &val) {
  return std::to_string(val);
}

/* \brief Convert container to string */
template <typename T, typename = typename std::enable_if<!std::is_arithmetic<T>::value>::type,
          typename = decltype(std::declval<T>().begin())>
inline std::string to_string_like(const T &container) {
  std::string str;
  str.reserve(1024);  // Assume strings are <1 KB
  str += "(";
  bool first = true;
  for (const auto &val : container) {
    if (!first) {
      str += ",";
    }
    str += to_string_like(val);
    first = false;
  }
  str += ")";
  return str;
}

#define TORCH_ERROR(...)                                              \
  do {                                                               \
    throw std::runtime_error(concat_strings( \
        __FILE__ ":", __LINE__, " in function ", __func__, ": ",     \
        concat_strings(__VA_ARGS__)));         \
  } while (false)

#define TORCH_CHECK_CUDA(expr)                                                 \
  do {                                                                        \
    const cudaError_t status_CHECK_CUDA = (expr);                        \
    if (status_CHECK_CUDA != cudaSuccess) {                              \
      TORCH_ERROR("CUDA Error: ", cudaGetErrorString(status_CHECK_CUDA)); \
    }                                                                         \
  } while (false)


/*! \brief Convert arguments to strings and concatenate */
template <typename... Ts>
inline std::string concat_strings(const Ts &...args) {
  std::string str;
  str.reserve(1024);  // Assume strings are <1 KB
  (..., (str += to_string_like(args)));
  return str;
}


template <typename T>
struct TypeInfo {

  using types = std::tuple<byte, int16, int32, int64, fp32, fp16, bf16, fp8e4m3, fp8e5m2, int8>;

  constexpr static size_t size = BitsNumber<T>::num_bits;
  constexpr static float max_finite_value = TypeExtrema<T>::max;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
  struct uint16 {
  uint4 u;
  uint4 v;
  uint4 s;
  uint4 t;
  };


  struct uint8 {
    uint4 u;
    uint4 v;
  };

  template <int BYTES>
  struct BytesToType {};

  template <>
  struct BytesToType<64> {
    using Type = uint16;
    static_assert(sizeof(Type) == 64);
  };

  template <>
  struct BytesToType<32> {
    using Type = uint8;
    static_assert(sizeof(Type) == 32);
  };

  template <>
  struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
  };

  template <>
  struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
  };

  template <>
  struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
  };

  template <>
  struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
  };

  template <>
  struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
  };

////////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename Elt_type, uint32_t NUM_ELT>
  struct Vec {
    enum { BYTES = NUM_ELT * sizeof(Elt_type) };

    using Vec_type = typename BytesToType<BYTES>::Type;
    using type = Elt_type;

    using Alias_type = union {
      Vec_type vec;
      Elt_type elt[NUM_ELT];
    };

    Alias_type data;
 
    __HOST_DEVICE__ Vec& operator=(const Vec& rhs) {
      data.vec = rhs.data.vec;
      return *this;
    }

    template <typename S>
    inline __device__ void to(Vec<S, NUM_ELT> &other) {  // NOLINT(*)
    #pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        other.data.elt[it] = S(this->data.elt[it]);
      }
    }

    template <typename Op>
    inline __device__ void assign(const Op &op) {
    #pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        this->data.elt[it] = op(it);
      }
    }

    // Pointer is cast to vector type
    inline __device__ void load_from(const void *base_ptr, size_t idx = 0) {
      this->data.vec = static_cast<const Vec_type *>(base_ptr)[idx];
    }

    // Pointer is cast to vector type
    inline __device__ void store_to(void *base_ptr, size_t idx = 0) const {
      static_cast<Vec_type *>(base_ptr)[idx] = this->data.vec;
    }

    // Pointer is cast to element type. Loads min(count, NUM_ELT)
    // elements and any remaining elements are set to zero.
    inline __device__ void load_from_elts(const void *base_ptr, size_t idx = 0,
                                          size_t count = NUM_ELT) {
      const Elt_type *elt_ptr = static_cast<const Elt_type *>(base_ptr) + idx;
      if (count < NUM_ELT || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0) {
  #pragma unroll
        for (int it = 0; it < NUM_ELT; it++) {
          this->data.elt[it] = (it < count ? elt_ptr[it] : Elt_type(0.f));
        }
      } else {
        this->load_from(elt_ptr);
      }
    }

    // Pointer is cast to element type. Stores min(count, NUM_ELT)
    // elements.
    inline __device__ void store_to_elts(void *base_ptr, size_t idx = 0,
                                        size_t count = NUM_ELT) const {
      Elt_type *elt_ptr = static_cast<Elt_type *>(base_ptr) + idx;
      if (count < NUM_ELT || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0) {
      #pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        if (it < count) {
          elt_ptr[it] = this->data.elt[it];
        }
      }
      } else {
        this->store_to(elt_ptr);
      }
        }

    inline __device__ void clear() {
    #pragma unroll
        for (int it = 0; it < NUM_ELT; it++) {
          this->data.elt[it] = Elt_type(0.f);
        }
      }
    };

////////////////////////////////////////////////////////////////////////////////////////////////////

  __device__ __forceinline__ float compute_scale_from_amax(float amax, float max_fp8,
                                                          bool force_pow_2_scales, float epsilon,
                                                          float value_for_inf) {
    // NOTE: NAN amax evaluates false for <, handled further down.
    if (amax < epsilon) {
      amax = epsilon;
    }

    float scale = 1.f;

    if (isinf(amax) || amax == 0.f || isnan(amax)) {
      return scale;
    }

    // Here we don't use "scale = max_fp8 / amax" because it has different results with/without
    // "--use_fast_math".
    // "__fdiv_rn" has the same behavior with "max_fp8 / amax" when not using fast math.
    scale = __fdiv_rn(max_fp8, amax);

    // The amax is too small that the scale becoming infinite in FP32. In other word,
    // the scale is not representable in FP32.
    if (isinf(scale)) {
      // use fp32 max to represent the scale
      scale = value_for_inf;
    }
    if (force_pow_2_scales) {
      uint32_t scale_bits = *reinterpret_cast<uint32_t *>(&scale);
      scale_bits &= 0xFF800000;
      // If the exponent was zero, we have a logic error.
      __builtin_assume(scale_bits != 0 || scale == 0.0);
      __builtin_assume(scale_bits != 0x80000000);
      scale = *reinterpret_cast<float *>(&scale_bits);
    }

    return scale;
  }

  // Calculate the quantization scale for an individual data element
  // given the amax(abs(tile)) value for a given quantization tile.
  //
  //
  // Arguments:
  // IType: data type of the tensor being quantized (float or bf16)
  // OType: quantized data type (e4m3 or e5m2)
  // amax: The evaluation of amax(abs(tile)) for the quantization tile.
  // eps: An epsilon used as a floor for amax.
  // pow_2_scaling: Whether to force the scale to be a power of 2.
  template <typename IType, typename OType>
  __device__ __forceinline__ float compute_scale_from_types(const float amax, const float eps,
                                                            const float pow_2_scaling) {
    constexpr float fp8_max = TypeInfo<OType>::max_finite_value;
    // NOTE: We're relying on compute_scale_from_amax to have behavior where it
    // clips the mantissa of the max_finite_value if power of 2 scaling applies.
    constexpr float value_for_inf = TypeInfo<IType>::max_finite_value;
    return compute_scale_from_amax(amax, fp8_max, pow_2_scaling, eps, value_for_inf);
  }



#define TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(dtype, type, ...) \
  switch (dtype) {                                             \
    case at::kFloat: {                                         \
      using type = float;                                      \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    case at::kHalf: {                                          \
      using type = fp16;                                       \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    case at::kBFloat16: {                                      \
      using type = bf16;                                       \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    case at::kFloat8_e5m2:                                      \
    case at::kFloat8_e4m3fn: {                                  \
      TORCH_ERROR("FP8 type not instantiated for input.");      \
    } break;                                                   \
    default:                                                   \
      TORCH_ERROR("Invalid type.");                             \
  }


#define TRANSFORMER_ENGINE_TYPE_SWITCH_8BIT(dtype, type, ...)    \
  switch (dtype) {                                               \
    case at::kFloat8_e5m2: {                                      \
      using type = fp8e5m2;                                      \
      { __VA_ARGS__ }                                            \
    } break;                                                     \
    case at::kFloat8_e4m3fn: {                                   \
      using type = fp8e4m3;                                      \
      { __VA_ARGS__ }                                            \
    } break;                                                     \
    case torch::kInt8: {                                         \
      using type = int8;                                         \
      { __VA_ARGS__ }                                            \
    } break;                                                     \
    default:                                                     \
      TORCH_ERROR("Invalid type.");                               \
  }

#define TRANSFORMER_ENGINE_SWITCH_CONDITION(CONDITION, FLAG, ...) \
  if (CONDITION) {                                                \
    constexpr bool FLAG = true;                                   \
    { __VA_ARGS__ }                                               \
  } else {                                                        \
    constexpr bool FLAG = false;                                  \
    { __VA_ARGS__ }                                               \
  }



}  //
} 

