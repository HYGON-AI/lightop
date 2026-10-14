// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cfloat>
#include <hip/hip_bf16.h>

namespace at {
namespace native {

typedef __hip_bfloat16 __nv_bfloat16;

constexpr int kMoePerGroupWaveSize = 64;
constexpr int kMoePerGroupWarpsPerBlock = 2;
constexpr int kMoePerGroupRoutedTopk = 7;

#if defined(__gfx938__)
__device__ inline float moe_per_group_exp(float value) {
  constexpr float kLog2E = 1.44269504089f;
  return __builtin_amdgcn_exp2f(value * kLog2E);
}
#else
__device__ inline float moe_per_group_exp(float value) {
  return __expf(value);
}
#endif

template <typename T>
__device__ inline uint16_t moe_per_group_bits(T value) {
  static_assert(sizeof(T) == sizeof(uint16_t));
  union {
    T value;
    uint16_t bits;
  } packed;
  packed.value = value;
  return packed.bits;
}

__device__ inline uint32_t moe_per_group_ordered_key(
    uint16_t score_bits,
    int expert_id) {
  const uint32_t ordered_score =
      (score_bits & 0x8000u)
          ? ((~static_cast<uint32_t>(score_bits)) & 0xffffu)
          : (static_cast<uint32_t>(score_bits) ^ 0x8000u);
  return
      (ordered_score << 16) |
      (0xffffu - static_cast<uint32_t>(expert_id));
}

template <typename T>
__device__ inline uint32_t moe_per_group_pack_key(
    float probability,
    T correction_bias,
    int expert_id) {
  // Match a native 16-bit sigmoid/add path: round sigmoid first, then add.
  const T rounded_probability = static_cast<T>(probability);
  const T rounded_choice_score = static_cast<T>(
      static_cast<float>(rounded_probability) +
      static_cast<float>(correction_bias));
  return moe_per_group_ordered_key(
      moe_per_group_bits(rounded_choice_score),
      expert_id);
}

template <int DPP_CTRL>
__device__ inline uint32_t moe_per_group_mov_dpp(uint32_t value) {
  return static_cast<uint32_t>(
      __builtin_amdgcn_mov_dpp(
          value, DPP_CTRL, 0xf, 0xf, true));
}

template <
    typename T,
    typename T_bias,
    int NUM_EXPERT_SLOTS,
    int NUM_EXPERT_GROUPS,
    int NUM_PAD_EXPERTS,
    int NUM_SHARED_EXPERTS,
    bool USE_PACKED_TOPK,
    int THREADS_PER_GROUP>
__global__ void moe_fuse_gate_per_gourp_topk_kernel(
    const T* input,
    const T_bias* bias,
    float* output,
    int32_t* indices,
    int64_t num_tokens,
    float routed_scaling_factor) {
  static_assert(
      NUM_EXPERT_GROUPS == 1 || NUM_EXPERT_GROUPS == 2,
      "Only one or two expert groups are supported");
  static_assert(
      NUM_EXPERT_SLOTS / NUM_EXPERT_GROUPS == 128,
      "Each expert group must contain 128 physical slots");
  static_assert(
      THREADS_PER_GROUP == 16 || THREADS_PER_GROUP == 32,
      "Each expert group must use 16 or 32 threads");
  static_assert(
      kMoePerGroupWaveSize % THREADS_PER_GROUP == 0,
      "The wave size must be divisible by threads per group");

  constexpr int kSlotsPerGroup =
      NUM_EXPERT_SLOTS / NUM_EXPERT_GROUPS;
  static_assert(
      NUM_PAD_EXPERTS % NUM_EXPERT_GROUPS == 0,
      "Padding expert slots must be distributed evenly across groups");
  constexpr int kPadExpertsPerGroup =
      NUM_PAD_EXPERTS / NUM_EXPERT_GROUPS;
  constexpr int kValuesPerThread =
      kSlotsPerGroup / THREADS_PER_GROUP;
  constexpr int kSubgroupsPerWave =
      kMoePerGroupWaveSize / THREADS_PER_GROUP;
  constexpr int kRowsPerWave =
      kSubgroupsPerWave / NUM_EXPERT_GROUPS;
  constexpr int kRowsPerBlock =
      kMoePerGroupWarpsPerBlock * kRowsPerWave;
  constexpr int kTotalTopk =
      NUM_EXPERT_GROUPS * kMoePerGroupRoutedTopk +
      NUM_SHARED_EXPERTS;
  constexpr float kNegativeInfinity = -FLT_MAX;

  const int thread_id = threadIdx.x;
  const int wave_id = thread_id / kMoePerGroupWaveSize;
  const int lane_id = thread_id % kMoePerGroupWaveSize;
  const int subgroup_id = lane_id / THREADS_PER_GROUP;
  const int lane_in_group = lane_id % THREADS_PER_GROUP;
  const int token_id =
      blockIdx.x * kRowsPerBlock + wave_id * kRowsPerWave +
      subgroup_id / NUM_EXPERT_GROUPS;
  const int expert_group = subgroup_id % NUM_EXPERT_GROUPS;

  if (token_id >= num_tokens) {
    return;
  }

  const int group_start = expert_group * kSlotsPerGroup;
  const int first_expert =
      group_start + lane_in_group * kValuesPerThread;
  const T* row_input = input + token_id * NUM_EXPERT_SLOTS;

  float probabilities[kValuesPerThread];
  float choice_scores[kValuesPerThread];
  uint32_t packed_keys[kValuesPerThread];

#pragma unroll
  for (int i = 0; i < kValuesPerThread; ++i) {
    const int expert_id = first_expert + i;
    const int expert_id_in_group = expert_id - group_start;
    const bool is_padding =
        kPadExpertsPerGroup > 0 &&
        expert_id_in_group >=
            kSlotsPerGroup - kPadExpertsPerGroup;
    if (is_padding) {
      probabilities[i] = 0.0f;
      choice_scores[i] = kNegativeInfinity;
      packed_keys[i] = 0;
    } else {
      const float logit = static_cast<float>(row_input[expert_id]);
      float probability;
      if constexpr (!USE_PACKED_TOPK) {
        // Match torch.sigmoid's FP32 exp path before any input-dtype
        // rounding. The gfx938 exp2 approximation can change near-tie top-k.
        probability = 1.0f / (1.0f + expf(-logit));
      } else {
        probability =
            1.0f / (1.0f + moe_per_group_exp(-logit));
      }
      // torch.sigmoid preserves the input dtype. Match that behavior before
      // applying a possibly-fp32 correction bias.
      if constexpr (sizeof(T) == sizeof(uint16_t)) {
        probability = static_cast<float>(static_cast<T>(probability));
      }
      if constexpr (USE_PACKED_TOPK) {
        probabilities[i] = probability;
        packed_keys[i] = moe_per_group_pack_key(
            probability, bias[expert_id], expert_id);
      } else {
        probabilities[i] = probability;
        choice_scores[i] =
            probability + static_cast<float>(bias[expert_id]);
      }
    }
  }

  float stored_probability = 0.0f;
  int stored_expert = 0;
  float group_probability_sum = 0.0f;

#pragma unroll
  for (int rank = 0; rank < kMoePerGroupRoutedTopk; ++rank) {
    int group_best_expert;
    if constexpr (USE_PACKED_TOPK) {
      uint32_t winner_key = packed_keys[0];
#pragma unroll
      for (int i = 1; i < kValuesPerThread; ++i) {
        winner_key =
            packed_keys[i] > winner_key
                ? packed_keys[i]
                : winner_key;
      }

      uint32_t other_key =
          moe_per_group_mov_dpp<0xb1>(winner_key);
      winner_key =
          other_key > winner_key ? other_key : winner_key;
      other_key = moe_per_group_mov_dpp<0x4e>(winner_key);
      winner_key =
          other_key > winner_key ? other_key : winner_key;
      other_key = __shfl_xor(
          winner_key, 4, THREADS_PER_GROUP);
      winner_key =
          other_key > winner_key ? other_key : winner_key;
      other_key = moe_per_group_mov_dpp<0x128>(winner_key);
      winner_key =
          other_key > winner_key ? other_key : winner_key;
      if constexpr (THREADS_PER_GROUP == 32) {
        other_key = __shfl_xor(
            winner_key, 16, THREADS_PER_GROUP);
        winner_key =
            other_key > winner_key ? other_key : winner_key;
      }

      group_best_expert =
          0xffff - static_cast<int>(winner_key & 0xffffu);

#pragma unroll
      for (int i = 0; i < kValuesPerThread; ++i) {
        if (packed_keys[i] == winner_key) {
          packed_keys[i] = 0;
        }
      }
    } else {
      float local_best_score = choice_scores[0];
      int local_best_expert = first_expert;

#pragma unroll
      for (int i = 1; i < kValuesPerThread; ++i) {
        const float candidate_score = choice_scores[i];
        const int candidate_expert = first_expert + i;
        if (candidate_score > local_best_score ||
            (candidate_score == local_best_score &&
             candidate_expert < local_best_expert)) {
          local_best_score = candidate_score;
          local_best_expert = candidate_expert;
        }
      }

      float group_best_score = local_best_score;
      group_best_expert = local_best_expert;

#pragma unroll
      for (int mask = THREADS_PER_GROUP / 2; mask > 0; mask >>= 1) {
        const float other_score =
            __shfl_xor(
                group_best_score, mask, THREADS_PER_GROUP);
        const int other_expert =
            __shfl_xor(
                group_best_expert, mask, THREADS_PER_GROUP);
        if (other_score > group_best_score ||
            (other_score == group_best_score &&
             other_expert < group_best_expert)) {
          group_best_score = other_score;
          group_best_expert = other_expert;
        }
      }
    }

    const int winner_group_offset =
        group_best_expert - group_start;
    const int winner_lane =
        winner_group_offset / kValuesPerThread;
    const int winner_local_offset =
        winner_group_offset % kValuesPerThread;
    float winner_probability = 0.0f;

    if (lane_in_group == winner_lane) {
      winner_probability = probabilities[winner_local_offset];
      if constexpr (!USE_PACKED_TOPK) {
        choice_scores[winner_local_offset] = kNegativeInfinity;
      }
    }
    winner_probability =
        __shfl(
            winner_probability,
            winner_lane,
            THREADS_PER_GROUP);
    group_probability_sum += winner_probability;

    if (lane_in_group == rank) {
      stored_probability = winner_probability;
      stored_expert = group_best_expert;
    }
  }

  if (lane_in_group < kMoePerGroupRoutedTopk) {
    const int output_rank =
        expert_group * kMoePerGroupRoutedTopk + lane_in_group;
    const int output_offset =
        token_id * kTotalTopk + output_rank;
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
      // Match normal PyTorch dtype semantics:
      //   weights / weights.sum(), optional scalar multiply, then float().
      const T rounded_sum =
          static_cast<T>(group_probability_sum);
      const T normalized_probability = static_cast<T>(
          stored_probability /
          static_cast<float>(rounded_sum));
      const T scaled_probability = static_cast<T>(
          static_cast<float>(normalized_probability) *
          routed_scaling_factor);
      output[output_offset] =
          static_cast<float>(scaled_probability);
    } else {
      output[output_offset] =
          stored_probability / group_probability_sum *
          routed_scaling_factor;
    }
    indices[output_offset] = stored_expert;
  }

  if constexpr (NUM_SHARED_EXPERTS > 0) {
    const bool writes_shared_expert =
        NUM_EXPERT_GROUPS == 1 || expert_group == 0;
    if (writes_shared_expert &&
        lane_in_group < NUM_SHARED_EXPERTS) {
      const int output_offset =
          token_id * kTotalTopk +
          NUM_EXPERT_GROUPS * kMoePerGroupRoutedTopk +
          lane_in_group;
      output[output_offset] = 1.0f;
      if constexpr (
          NUM_EXPERT_GROUPS == 2 &&
          NUM_PAD_EXPERTS == NUM_SHARED_EXPERTS) {
        indices[output_offset] =
            (lane_in_group + 1) * kSlotsPerGroup - 1;
      } else {
        indices[output_offset] =
            NUM_EXPERT_SLOTS + lane_in_group;
      }
    }
  }
}

template <
    typename T,
    typename T_bias,
    int NUM_EXPERT_SLOTS,
    int NUM_EXPERT_GROUPS,
    int NUM_PAD_EXPERTS,
    int NUM_SHARED_EXPERTS,
    bool USE_PACKED_TOPK,
    int THREADS_PER_GROUP>
void launch_moe_fuse_gate_per_gourp_topk(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output,
    torch::Tensor& indices,
    float routed_scaling_factor,
    cudaStream_t stream) {
  constexpr int kSubgroupsPerWave =
      kMoePerGroupWaveSize / THREADS_PER_GROUP;
  constexpr int kRowsPerWave =
      kSubgroupsPerWave / NUM_EXPERT_GROUPS;
  constexpr int kRowsPerBlock =
      kMoePerGroupWarpsPerBlock * kRowsPerWave;
  const int64_t num_tokens = input.size(0);
  const int64_t num_blocks =
      (num_tokens + kRowsPerBlock - 1) / kRowsPerBlock;

  moe_fuse_gate_per_gourp_topk_kernel<
      T,
      T_bias,
      NUM_EXPERT_SLOTS,
      NUM_EXPERT_GROUPS,
      NUM_PAD_EXPERTS,
      NUM_SHARED_EXPERTS,
      USE_PACKED_TOPK,
      THREADS_PER_GROUP>
      <<<num_blocks,
         kMoePerGroupWaveSize * kMoePerGroupWarpsPerBlock,
         0,
         stream>>>(
          reinterpret_cast<const T*>(input.data_ptr()),
          reinterpret_cast<const T_bias*>(bias.data_ptr()),
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          num_tokens,
          routed_scaling_factor);
}

template <
    typename T,
    typename T_bias,
    bool USE_PACKED_TOPK,
    int THREADS_PER_GROUP>
void dispatch_moe_fuse_gate_per_gourp_topk_layout(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output,
    torch::Tensor& indices,
    bool is_30b,
    int64_t num_shared_experts,
    float routed_scaling_factor,
    cudaStream_t stream) {
  if (is_30b) {
    if (num_shared_experts == 0) {
      launch_moe_fuse_gate_per_gourp_topk<
          T,
          T_bias,
          128,
          1,
          0,
          0,
          USE_PACKED_TOPK,
          THREADS_PER_GROUP>(
          input,
          bias,
          output,
          indices,
          routed_scaling_factor,
          stream);
    } else {
      launch_moe_fuse_gate_per_gourp_topk<
          T,
          T_bias,
          128,
          1,
          0,
          1,
          USE_PACKED_TOPK,
          THREADS_PER_GROUP>(
          input,
          bias,
          output,
          indices,
          routed_scaling_factor,
          stream);
    }
  } else {
    if (num_shared_experts == 0) {
      launch_moe_fuse_gate_per_gourp_topk<
          T,
          T_bias,
          256,
          2,
          2,
          0,
          USE_PACKED_TOPK,
          THREADS_PER_GROUP>(
          input,
          bias,
          output,
          indices,
          routed_scaling_factor,
          stream);
    } else {
      launch_moe_fuse_gate_per_gourp_topk<
          T,
          T_bias,
          256,
          2,
          2,
          2,
          USE_PACKED_TOPK,
          THREADS_PER_GROUP>(
          input,
          bias,
          output,
          indices,
          routed_scaling_factor,
          stream);
    }
  }
}

inline bool moe_per_group_use_16_threads_fp16(
    bool is_30b,
    int64_t num_tokens) {
  // Use the same crossover on gfx936 and gfx938. The measured gfx938
  // crossover is 4096 tokens for 30B and 2048 tokens for 300B.
  return num_tokens >= (is_30b ? 4096 : 2048);
}

std::vector<torch::Tensor> moe_fuse_gate_per_gourp_topk(
    torch::Tensor& input,
    torch::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_per_group,
    int64_t num_pad_experts,
    int64_t num_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  TORCH_CHECK(
      input.is_cuda() && bias.is_cuda(),
      "input and bias must be CUDA tensors");
  TORCH_CHECK(
      input.dim() == 2,
      "input must have shape [num_tokens, num_expert_slots]");
  TORCH_CHECK(
      bias.dim() == 1 && bias.numel() == input.size(1),
      "bias must have one element per expert slot");
  TORCH_CHECK(
      input.is_contiguous() && bias.is_contiguous(),
      "input and bias must be contiguous");
  TORCH_CHECK(
      input.device() == bias.device(),
      "input and bias must be on the same device");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  const bool is_30b =
      input.size(1) == 128 &&
      num_expert_group == 1 &&
      topk_per_group == 7 &&
      num_pad_experts == 0 &&
      (num_shared_experts == 0 || num_shared_experts == 1);
  const bool is_300b =
      input.size(1) == 256 &&
      num_expert_group == 2 &&
      topk_per_group == 7 &&
      num_pad_experts == 2 &&
      (num_shared_experts == 0 || num_shared_experts == 2);

  TORCH_CHECK(
      is_30b || is_300b,
      "moe_fuse_gate_per_gourp_topk supports SparkV1 30B "
      "(128 slots, group=1, topk_per_group=7, pad=0, shared=0/1) "
      "or SparkV1 300B "
      "(256 slots, group=2, topk_per_group=7, pad=2, shared=0/2)");

  const int64_t total_topk =
      num_expert_group * topk_per_group + num_shared_experts;
  auto output = torch::empty(
      {input.size(0), total_topk},
      input.options().dtype(torch::kFloat32));
  auto indices = torch::empty(
      {input.size(0), total_topk},
      input.options().dtype(torch::kInt32));

  if (input.size(0) == 0) {
    return {output, indices};
  }

  const float output_scale =
      apply_routed_scaling_factor_on_output &&
              routed_scaling_factor > 0.0
          ? static_cast<float>(routed_scaling_factor)
          : 1.0f;
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream();
  bool dispatched = false;

  if (input.scalar_type() == at::kBFloat16) {
    if (bias.scalar_type() == at::kFloat) {
      dispatch_moe_fuse_gate_per_gourp_topk_layout<
          __nv_bfloat16, float, false, 32>(
          input,
          bias,
          output,
          indices,
          is_30b,
          num_shared_experts,
          output_scale,
          stream);
      dispatched = true;
    } else if (bias.scalar_type() == at::kBFloat16) {
      dispatch_moe_fuse_gate_per_gourp_topk_layout<
          __nv_bfloat16, __nv_bfloat16, true, 32>(
          input,
          bias,
          output,
          indices,
          is_30b,
          num_shared_experts,
          output_scale,
          stream);
      dispatched = true;
    }
  } else if (
      input.scalar_type() == at::kHalf &&
      bias.scalar_type() == at::kHalf) {
    if (moe_per_group_use_16_threads_fp16(
            is_30b, input.size(0))) {
      dispatch_moe_fuse_gate_per_gourp_topk_layout<
          half, half, true, 16>(
          input,
          bias,
          output,
          indices,
          is_30b,
          num_shared_experts,
          output_scale,
          stream);
    } else {
      dispatch_moe_fuse_gate_per_gourp_topk_layout<
          half, half, true, 32>(
          input,
          bias,
          output,
          indices,
          is_30b,
          num_shared_experts,
          output_scale,
          stream);
    }
    dispatched = true;
  } else if (input.scalar_type() == at::kFloat) {
    if (bias.scalar_type() == at::kFloat) {
      dispatch_moe_fuse_gate_per_gourp_topk_layout<
          float, float, false, 32>(
          input,
          bias,
          output,
          indices,
          is_30b,
          num_shared_experts,
          output_scale,
          stream);
      dispatched = true;
    } else if (bias.scalar_type() == at::kBFloat16) {
      dispatch_moe_fuse_gate_per_gourp_topk_layout<
          float, __nv_bfloat16, false, 32>(
          input,
          bias,
          output,
          indices,
          is_30b,
          num_shared_experts,
          output_scale,
          stream);
      dispatched = true;
    } else if (bias.scalar_type() == at::kHalf) {
      dispatch_moe_fuse_gate_per_gourp_topk_layout<
          float, half, false, 32>(
          input,
          bias,
          output,
          indices,
          is_30b,
          num_shared_experts,
          output_scale,
          stream);
      dispatched = true;
    }
  }

  TORCH_CHECK(
      dispatched,
      "Unsupported input/bias dtype combination for "
      "moe_fuse_gate_per_gourp_topk");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, indices};
}

}  // namespace native
}  // namespace at
