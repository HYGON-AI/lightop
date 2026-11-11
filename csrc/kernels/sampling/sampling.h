// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <exception>
#include <iostream>
#include <sstream>

namespace lightop {
namespace sampling {

class Error : public std::exception {
private:
  std::string message_;

public:
  Error(const std::string& func, const std::string& file, int line, const std::string& message) {
    std::ostringstream oss;
    oss << "Error in function '" << func << "' "
        << "at " << file << ":" << line << ": " << message;
    message_ = oss.str();
  }

  virtual const char* what() const noexcept override { return message_.c_str(); }
};

#define SAMPLING_ERROR(message) throw Error(__FUNCTION__, __FILE__, __LINE__, message)

#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                              \
    case 16: {                                                             \
      constexpr size_t ALIGNED_VEC_SIZE = 16;                              \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 8: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 8;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 4: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 4;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 2: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 2;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 1: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default: {                                                             \
      std::ostringstream err_msg;                                          \
      err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;     \
      SAMPLING_ERROR(err_msg.str());                                       \
    }                                                                      \
  }

#define DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, ...) \
  if (deterministic) {                                            \
    constexpr bool DETERMINISTIC = true;                          \
    __VA_ARGS__                                                   \
  } else {                                                        \
    constexpr bool DETERMINISTIC = false;                         \
    __VA_ARGS__                                                   \
  }

#define DISPATCH_DLPACK_IDTYPE_TO_CTYPE(dlpack_dtype, c_type, ...)                       \
  [&]() -> bool {                                                                        \
    switch (encode_dlpack_dtype(dlpack_dtype)) {                                         \
      _DISPATCH_CASE_I32(c_type, __VA_ARGS__)                                            \
      _DISPATCH_CASE_I64(c_type, __VA_ARGS__)                                            \
      default:                                                                           \
        TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " failed to dispatch data type " \
                              << (dlpack_dtype).code << " " << (dlpack_dtype).bits;      \
        return false;                                                                    \
    }                                                                                    \
  }()

void top_k_sampling_from_probs(const torch::Tensor& probs, torch::Tensor& output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset);

void top_p_sampling_from_probs(const torch::Tensor& probs, torch::Tensor& output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_p_arr, float top_p_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset);

void top_k_top_p_sampling_from_probs(const torch::Tensor& probs, torch::Tensor& output,
                                     std::optional<torch::Tensor> maybe_indices,
                                     std::optional<torch::Tensor> maybe_top_k_arr, int64_t top_k_val,
                                     std::optional<torch::Tensor> maybe_top_p_arr, float top_p_val,
                                     bool deterministic, uint64_t philox_seed,
                                     uint64_t philox_offset);

void top_k_first_top_p_sampling_from_probs(
    const torch::Tensor& probs, torch::Tensor& output,
    std::optional<torch::Tensor> maybe_indices,
    std::optional<torch::Tensor> maybe_top_k_arr, int64_t top_k_val,
    std::optional<torch::Tensor> maybe_top_p_arr, float top_p_val,
    bool deterministic, uint64_t philox_seed, uint64_t philox_offset);

}
}
