// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Retained GLM prefill K8 gather; common gather APIs are unchanged.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

namespace lightop_glm53_ep_gather {
template <typename T, int N> struct alignas(sizeof(T) * N) AlignedVector {
  T val[N];
};

// Unchanged arithmetic/body from B2 TU SHA256 2dc6f5d5...; only symbol renamed.
template <typename scalar_t, typename TopKType, int K_MAX>
__global__ void
original_control(const scalar_t *__restrict__ input_tensor,
                 const TopKType *__restrict__ topk_ids,
                 const float *__restrict__ topk_weights,
                 const int *__restrict__ src_indices, // inverm
                 const int *__restrict__ expert_map,
                 scalar_t *__restrict__ output_tensor, const int hidden_size,
                 const int topk_num, const int expert_map_size,
                 const bool has_expert_map, const int num_tokens) {

  constexpr int VEC_SIZE = 8;
  using VecType = AlignedVector<scalar_t, VEC_SIZE>;

  const int col_offset_base = blockIdx.x * blockDim.x * VEC_SIZE;
  const int tid = threadIdx.x;
  const int feat_idx = col_offset_base + tid * VEC_SIZE;
  if (feat_idx >= hidden_size)
    return;

  for (int token_idx = blockIdx.y; token_idx < num_tokens;
       token_idx += gridDim.y) {
    float acc[VEC_SIZE] = {0.0f};
#pragma unroll
    for (int k = 0; k < K_MAX; ++k) {
      if (k >= topk_num) {
        continue;
      }
      size_t flat_idx = static_cast<size_t>(token_idx) * topk_num + k;
      int64_t global_expert_id = static_cast<int64_t>(topk_ids[flat_idx]);
      bool is_valid = global_expert_id >= 0;
      if (is_valid && has_expert_map) {
        if (global_expert_id >= static_cast<int64_t>(expert_map_size)) {
          is_valid = false;
        } else {
          int expert_id = expert_map[global_expert_id];
          is_valid = expert_id >= 0;
        }
      }
      int src_idx = src_indices[flat_idx];
      if (is_valid && src_idx >= 0) {
        float w = topk_weights[flat_idx];

        size_t src_offset =
            static_cast<size_t>(src_idx) * hidden_size + feat_idx;
        const scalar_t *src_ptr = input_tensor + src_offset;

        VecType loaded_vec = *reinterpret_cast<const VecType *>(src_ptr);

#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          acc[i] += static_cast<float>(loaded_vec.val[i]) * w;
        }
      }
    }

    VecType out_vec;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      out_vec.val[i] = static_cast<scalar_t>(acc[i]);
    }

    size_t out_offset = static_cast<size_t>(token_idx) * hidden_size + feat_idx;
    VecType *out_vec_ptr =
        reinterpret_cast<VecType *>(output_tensor + out_offset);
    *out_vec_ptr = out_vec;
  }
}

using BFloat16 = c10::BFloat16;
using Vec = AlignedVector<BFloat16, 8>;
struct Fetched {
  Vec values;
  float weight;
  bool valid;
};

__device__ __forceinline__ Fetched fetch(const BFloat16 *__restrict__ input,
                                         const int64_t *__restrict__ ids,
                                         const float *__restrict__ weights,
                                         const int32_t *__restrict__ src,
                                         int token, int k, int hidden,
                                         int feature) {
  const size_t flat = static_cast<size_t>(token) * 8 + k;
  Fetched item{};
  const int32_t row = src[flat];
  // No-map native semantics do NOT impose an upper expert-id bound.
  item.valid = ids[flat] >= 0 && row >= 0;
  if (item.valid) {
    item.weight = weights[flat];
    item.values = *reinterpret_cast<const Vec *>(
        input + static_cast<size_t>(row) * hidden + feature);
  }
  return item;
}

template <bool Prefetch>
__global__ void specialized_k8_no_map(const BFloat16 *__restrict__ input,
                                      const int64_t *__restrict__ ids,
                                      const float *__restrict__ weights,
                                      const int32_t *__restrict__ src,
                                      BFloat16 *__restrict__ output, int hidden,
                                      int tokens) {
  const int feature = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (feature >= hidden)
    return;
  for (int token = blockIdx.y; token < tokens; token += gridDim.y) {
    float acc[8] = {0.0f};
    if constexpr (Prefetch) {
      Fetched current =
          fetch(input, ids, weights, src, token, 0, hidden, feature);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        Fetched next{};
        if (k + 1 < 8)
          next = fetch(input, ids, weights, src, token, k + 1, hidden, feature);
        if (current.valid) {
#pragma unroll
          for (int i = 0; i < 8; ++i)
            acc[i] = fmaf(static_cast<float>(current.values.val[i]),
                          current.weight, acc[i]);
        }
        current = next;
      }
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const Fetched current =
            fetch(input, ids, weights, src, token, k, hidden, feature);
        if (current.valid) {
#pragma unroll
          for (int i = 0; i < 8; ++i)
            acc[i] = fmaf(static_cast<float>(current.values.val[i]),
                          current.weight, acc[i]);
        }
      }
    }
    Vec result;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      result.val[i] = static_cast<BFloat16>(acc[i]);
    *reinterpret_cast<Vec *>(output + static_cast<size_t>(token) * hidden +
                             feature) = result;
  }
}

void run(torch::Tensor a, torch::Tensor ids, torch::Tensor weights,
         torch::Tensor src, torch::Tensor output, int64_t mode) {
  TORCH_CHECK(mode >= 0 && mode <= 2, "mode: 0 control, 1 K8, 2 K8 one-ahead");
  TORCH_CHECK(a.is_cuda() && ids.is_cuda() && weights.is_cuda() &&
                  src.is_cuda() && output.is_cuda(),
              "all operands must be device tensors");
  TORCH_CHECK(a.device() == ids.device() && a.device() == weights.device() &&
                  a.device() == src.device() && a.device() == output.device(),
              "all operands must share the captured device");
  TORCH_CHECK(
      a.scalar_type() == at::kBFloat16 &&
          output.scalar_type() == at::kBFloat16 &&
          ids.scalar_type() == at::kLong &&
          weights.scalar_type() == at::kFloat && src.scalar_type() == at::kInt,
      "requires BF16 input/output, int64 ids, FP32 weights, int32 inv_perm");
  TORCH_CHECK(
      a.is_contiguous() && output.is_contiguous() && ids.is_contiguous() &&
          weights.is_contiguous() && src.is_contiguous(),
      "diagnostic supports only the native contiguous eligibility path");
  TORCH_CHECK(a.dim() == 2 && output.dim() == 2 && ids.dim() == 2 &&
                  weights.sizes() == ids.sizes() &&
                  src.sizes() == ids.sizes() && ids.size(1) == 8 &&
                  ids.size(0) == output.size(0) && a.size(1) == output.size(1),
              "requires exact K8 captured shapes");
  TORCH_CHECK(a.size(1) > 0 && a.size(1) % 8 == 0 && a.size(1) <= INT32_MAX &&
                  output.size(0) <= INT32_MAX,
              "unsupported hidden or token range");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0,
              "16-byte vector alignment is required");
  TORCH_CHECK(a.data_ptr() != output.data_ptr(),
              "native restrict contract forbids alias");
  if (output.size(0) == 0)
    return;
  c10::cuda::CUDAGuard guard(a.device());
  const int hidden = static_cast<int>(a.size(1));
  const int tokens = static_cast<int>(output.size(0));
  int threads = 256;
  int grid_x = (hidden + threads * 8 - 1) / (threads * 8);
  if (hidden == 7168) {
    threads = 128;
    grid_x = 7;
  } else if (tokens >= 256 && tokens <= 512 && hidden == 4096) {
    threads = 128;
    grid_x = 4;
  }
  const dim3 grid(grid_x, tokens), block(threads);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto *input = a.data_ptr<BFloat16>();
  const auto *topk_ids = ids.data_ptr<int64_t>();
  const auto *topk_weights = weights.data_ptr<float>();
  const auto *inv_perm = src.data_ptr<int32_t>();
  auto *out = output.data_ptr<BFloat16>();
  if (mode == 0) {
    original_control<BFloat16, int64_t, 8>
        <<<grid, block, 0, stream>>>(input, topk_ids, topk_weights, inv_perm,
                                     nullptr, out, hidden, 8, 0, false, tokens);
  } else if (mode == 1) {
    specialized_k8_no_map<false><<<grid, block, 0, stream>>>(
        input, topk_ids, topk_weights, inv_perm, out, hidden, tokens);
  } else {
    specialized_k8_no_map<true><<<grid, block, 0, stream>>>(
        input, topk_ids, topk_weights, inv_perm, out, hidden, tokens);
  }
  // No host synchronization or data-dependent device read in the launcher.
  const auto error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, hipGetErrorString(error));
}
} // namespace lightop_glm53_ep_gather
