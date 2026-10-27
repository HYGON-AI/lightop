// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>

namespace at {
namespace native {
namespace {

struct IntegerInput {
  const void* data;
  bool wide;

  __device__ __forceinline__ int64_t operator[](int64_t i) const {
    return wide ? static_cast<const int64_t*>(data)[i] : static_cast<const int32_t*>(data)[i];
  }
};

IntegerInput integer_input(const Tensor& x) {
  return {x.data_ptr(), x.scalar_type() == at::kLong};
}

// Eight heads for each of two Engram tables. A lane computes all three
// n-gram hashes, reusing its predecessor IDs and rolling XOR.
__global__ void engram_hash_ids_hcu_kernel(
    IntegerInput input_ids, IntegerInput positions, IntegerInput history,
    const int64_t* __restrict__ row, const int64_t* __restrict__ starts,
    const int64_t* __restrict__ token_map, const int64_t* __restrict__ multipliers,
    const int64_t* __restrict__ primes, const int64_t* __restrict__ offsets,
    int64_t* __restrict__ hashes, int64_t* __restrict__ tokens,
    int64_t num_tokens, int64_t num_real, int64_t pad_id,
    int64_t image_token_id, int64_t mm_pad_shift) {
  const int head = threadIdx.x % 8;
  const int layer = (threadIdx.x / 8) % 2;
  for (int64_t t = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 16;
       t < num_tokens; t += int64_t(gridDim.x) * blockDim.x / 16) {
    const bool live = t < num_real;
    const int64_t request = live ? row[t] : 0;
    const int64_t offset = live ? t - starts[request] : 0;
    const int64_t position = live ? positions[t] : 0;
    bool image_seen = false;
    int64_t rolling = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      int64_t token = 0;
      if (live) {
        token = s <= offset ? input_ids[t - s] : history[request * 3 + 3 + offset - s];
      }
      if (image_token_id >= 0) {
        if (token >= mm_pad_shift) token = image_token_id;
        image_seen |= token == image_token_id;
      }
      if (layer == 0 && head == 0) tokens[t * 4 + s] = token;
      const bool blocked = !live || position < s || image_seen;
      const int64_t compressed = blocked ? pad_id : token_map[token];
      rolling ^= compressed * multipliers[layer * 4 + s];
      if (s > 0) {
        const int column = layer * 24 + (s - 1) * 8 + head;
        hashes[t * 48 + column] = rolling % primes[column] + offsets[column];
      }
    }
  }
}

}  // namespace

std::vector<Tensor> engram_hash_ids_hcu(
    const Tensor& input_ids, const Tensor& positions, const Tensor& history,
    const Tensor& row, const Tensor& starts, const Tensor& token_map,
    const Tensor& multipliers, const Tensor& primes, const Tensor& offsets,
    int64_t num_real, int64_t pad_id, int64_t image_token_id, int64_t mm_pad_shift) {
  for (const auto* x : {&input_ids, &positions, &history, &row, &starts, &token_map,
                        &multipliers, &primes, &offsets}) {
    TORCH_CHECK(x->is_cuda() && x->device() == input_ids.device() &&
                    x->is_contiguous() && !x->is_neg(),
                "Engram inputs must be contiguous tensors on one GPU");
  }
  for (const auto* x : {&input_ids, &positions, &history}) {
    TORCH_CHECK(x->scalar_type() == at::kInt || x->scalar_type() == at::kLong,
                "IDs, positions and history must be int32 or int64");
  }
  for (const auto* x : {&row, &starts, &token_map, &multipliers, &primes, &offsets}) {
    TORCH_CHECK(x->scalar_type() == at::kLong, "Engram hash metadata must be int64");
  }
  TORCH_CHECK(input_ids.dim() == 1 && positions.sizes() == input_ids.sizes(),
              "IDs and positions must be [T]");
  TORCH_CHECK(num_real >= 0 && num_real <= input_ids.numel() &&
                  row.dim() == 1 && row.numel() == num_real,
              "row must cover exactly the real token prefix");
  TORCH_CHECK(history.dim() == 2 && history.size(1) == 3 && starts.dim() == 1 &&
                  starts.numel() == history.size(0) && (num_real == 0 || starts.numel() > 0),
              "history must be [B,3] and starts [B]");
  TORCH_CHECK(multipliers.sizes() == at::IntArrayRef({2, 4}) &&
                  primes.sizes() == at::IntArrayRef({2, 3, 8}) &&
                  offsets.sizes() == at::IntArrayRef({2, 24}),
              "Engram HCU hash supports two tables, 4-grams and eight heads");
  TORCH_CHECK(token_map.dim() == 1 && token_map.numel() > 0 && pad_id >= 0 &&
                  image_token_id >= -1 && image_token_id < token_map.numel(),
              "invalid token map or padding/image ID");
  const c10::cuda::CUDAGuard guard(input_ids.device());
  const auto options = input_ids.options().dtype(at::kLong);
  auto hashes = at::empty({input_ids.numel(), 2, 24}, options);
  auto tokens = at::empty({input_ids.numel(), 4}, options);
  if (input_ids.numel() == 0) return {hashes, tokens};
  const int blocks = std::min<int64_t>((input_ids.numel() + 15) / 16, 65535);
  engram_hash_ids_hcu_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      integer_input(input_ids), integer_input(positions), integer_input(history),
      row.data_ptr<int64_t>(), starts.data_ptr<int64_t>(), token_map.data_ptr<int64_t>(),
      multipliers.data_ptr<int64_t>(), primes.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
      hashes.data_ptr<int64_t>(), tokens.data_ptr<int64_t>(), input_ids.numel(), num_real,
      pad_id, image_token_id, mm_pad_shift);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {hashes, tokens};
}

}  // namespace native
}  // namespace at
