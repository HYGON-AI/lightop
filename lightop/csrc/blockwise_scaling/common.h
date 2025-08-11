/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

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
using fp8e4m3 = __hip_fp8_e4m3;
using fp8e5m2 = __hip_fp8_e5m2;


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
TRANSFORMER_ENGINE_TYPE_NAME(__hip_fp8_e4m3)
TRANSFORMER_ENGINE_TYPE_NAME(__hip_fp8_e5m2)
TRANSFORMER_ENGINE_TYPE_NAME(int8_t)
#undef TRANSFORMER_ENGINE_TYPE_NAME


template <typename T>
struct TypeExtrema;

template <>
struct TypeExtrema<fp8e4m3> {
  static constexpr float max = 448.0f;
};

template <>
struct TypeExtrema<int8> {
  static constexpr float max = 127.0f;
};

template <>
struct TypeExtrema<fp8e5m2> {
  static constexpr float max = 57344.0f;
};

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


#define TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(dtype, type, ...) \
  switch (dtype) {                                             \
    case at::kFloat: {                                         \
      using type = float;                                      \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    case at::kBFloat16: {                                      \
      using type = bf16;                                       \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    case at::kFloat8_e5m2:                                      \
    case at::kFloat8_e4m3fn: {                                     \
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

} 
} 

