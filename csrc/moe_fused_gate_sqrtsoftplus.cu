#include <cfloat>
#include <cstdint>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#if defined(__gfx938__)
constexpr float LOG2_E_VAL = 1.44269504089f;
__device__ inline float exp_func(float x) {
  return __builtin_amdgcn_exp2f(x * LOG2_E_VAL);
}
#else
__device__ inline float exp_func(float x) {
  return __expf(x);
}
#endif

namespace at {
namespace native {

#ifndef USE_ROCM
#define LIGHTOP_SHFL_XOR_WIDTH(var, lane_mask, width) \
  __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
#define LIGHTOP_SHFL_BCAST_WIDTH(var, src_lane, width) \
  __shfl_sync(uint32_t(-1), var, src_lane, width)
#else
#define LIGHTOP_SHFL_XOR_WIDTH(var, lane_mask, width) \
  __shfl_xor(var, lane_mask, width)
#define LIGHTOP_SHFL_BCAST_WIDTH(var, src_lane, width) \
  __shfl(var, src_lane, width)
#endif

static constexpr int WARP_SIZE = 64;
static constexpr int MAX_TOPK = 16;

template <typename T, int N, int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
  T data[N];

 public:
  __device__ T& operator[](int index) { return data[index]; }
  __device__ const T& operator[](int index) const { return data[index]; }
};

template <int VPT_, int NUM_EXPERTS_, int THREADS_PER_ROW_,
          int ROWS_PER_WARP_, int ROWS_PER_CTA_, int WARPS_PER_CTA_>
struct KernelParams {
  static constexpr int VPT = VPT_;
  static constexpr int NUM_EXPERTS = NUM_EXPERTS_;
  static constexpr int THREADS_PER_ROW = THREADS_PER_ROW_;
  static constexpr int ROWS_PER_WARP = ROWS_PER_WARP_;
  static constexpr int ROWS_PER_CTA = ROWS_PER_CTA_;
  static constexpr int WARPS_PER_CTA = WARPS_PER_CTA_;
};

__device__ __forceinline__ float sqrtsoftplus_score(float x) {
  return sqrtf(log1pf(exp_func(x)));
}

template <int VPT_LOCAL, int NUM_EXPERTS, int TOPK, typename Params>
__global__ void moe_fused_gate_sqrtsoftplus_kernel_group1(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int32_t* __restrict__ indices,
    int64_t num_rows,
    int64_t num_fused_shared_experts,
    bool renormalize,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  constexpr float NEG_INF = -FLT_MAX;
  static_assert(NUM_EXPERTS % VPT_LOCAL == 0,
                "NUM_EXPERTS must be divisible by VPT_LOCAL");
  constexpr int EFFECTIVE_THREADS = NUM_EXPERTS / VPT_LOCAL;
  static_assert(EFFECTIVE_THREADS <= Params::THREADS_PER_ROW,
                "THREADS_PER_ROW does not cover all experts");
  static_assert(Params::THREADS_PER_ROW <= WARP_SIZE,
                "one token must fit in a single wave");

  const int tidx = threadIdx.x;
  const int group_row = tidx / Params::THREADS_PER_ROW;
  const int row = blockIdx.x * Params::ROWS_PER_CTA + group_row;
  if (row >= num_rows) return;

  const int lane = tidx % Params::THREADS_PER_ROW;
  const int start_col = lane * VPT_LOCAL;
  const int topk_routed = TOPK - static_cast<int>(num_fused_shared_experts);

  float choice_scores[VPT_LOCAL];
  float route_scores[VPT_LOCAL];

  if (lane < EFFECTIVE_THREADS) {
    using Vec = AlignedArray<float, VPT_LOCAL>;
    const float* row_input = input + row * NUM_EXPERTS + start_col;
    const float* row_bias = bias + start_col;
    Vec input_vec = *reinterpret_cast<const Vec*>(row_input);
    Vec bias_vec = *reinterpret_cast<const Vec*>(row_bias);

#pragma unroll
    for (int i = 0; i < VPT_LOCAL; ++i) {
      const float score = sqrtsoftplus_score(input_vec[i]);
      route_scores[i] = score;
      choice_scores[i] = score + bias_vec[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < VPT_LOCAL; ++i) {
      route_scores[i] = 0.0f;
      choice_scores[i] = NEG_INF;
    }
  }

  float stored_weight = 0.0f;
  int stored_expert = 0;
  float routed_sum = 0.0f;

  for (int k = 0; k < TOPK; ++k) {
    if (k >= topk_routed) break;

    float local_max = NEG_INF;
    int local_expert = (lane < EFFECTIVE_THREADS) ? start_col : 0;

#pragma unroll
    for (int i = 0; i < VPT_LOCAL; ++i) {
      if (choice_scores[i] > local_max) {
        local_max = choice_scores[i];
        local_expert = start_col + i;
      }
    }

    float best_val = local_max;
    int best_expert = local_expert;
#pragma unroll
    for (int mask = Params::THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
      const float other_val =
          LIGHTOP_SHFL_XOR_WIDTH(best_val, mask, Params::THREADS_PER_ROW);
      const int other_expert =
          LIGHTOP_SHFL_XOR_WIDTH(best_expert, mask, Params::THREADS_PER_ROW);
      if (other_val > best_val) {
        best_val = other_val;
        best_expert = other_expert;
      }
    }

    const int winner_lane = best_expert / VPT_LOCAL;
    const int winner_offset = best_expert % VPT_LOCAL;
    float winner_route_score = 0.0f;
    if (lane == winner_lane) {
      choice_scores[winner_offset] = NEG_INF;
      winner_route_score = route_scores[winner_offset];
    }

    const float selected_weight = LIGHTOP_SHFL_BCAST_WIDTH(
        winner_route_score, winner_lane, Params::THREADS_PER_ROW);

    routed_sum += selected_weight;
    if (lane == k) {
      stored_weight = selected_weight;
      stored_expert = best_expert;
    }
  }

  if (num_fused_shared_experts > 0 && lane >= topk_routed && lane < TOPK) {
    const int shared_offset = lane - topk_routed;
    stored_weight = routed_sum / routed_scaling_factor;
    stored_expert = NUM_EXPERTS + shared_offset;
  }

  if (lane < TOPK) {
    const int out_idx = row * TOPK + lane;
    const float norm = (renormalize && routed_sum > 0.0f) ? routed_sum : 1.0f;
    const float scale =
        apply_routed_scaling_factor_on_output ? routed_scaling_factor : 1.0f;
    output[out_idx] = stored_weight / norm * scale;
    indices[out_idx] = stored_expert;
  }
}

#define LAUNCH_SQRTSOFTPLUS_GATE(                                                \
    TOPK_VAL, NUM_EXPERTS_VAL, THREADS_PER_ROW_VAL, VPT_VAL, N_WARP_VAL)          \
  do {                                                                              \
    constexpr int TOPK_C = TOPK_VAL;                                                \
    constexpr int NUM_EXPERTS_C = NUM_EXPERTS_VAL;                                  \
    constexpr int THREADS_PER_ROW_C = THREADS_PER_ROW_VAL;                          \
    constexpr int VPT_C = VPT_VAL;                                                  \
    constexpr int N_WARP_C = N_WARP_VAL;                                            \
    constexpr int ROWS_PER_WARP_C = WARP_SIZE / THREADS_PER_ROW_C;                  \
    constexpr int ROWS_PER_CTA_C = N_WARP_C * ROWS_PER_WARP_C;                      \
    dim3 block_dim(WARP_SIZE * N_WARP_C);                                           \
    int64_t grid = (num_rows + ROWS_PER_CTA_C - 1) / ROWS_PER_CTA_C;                \
    using ParamsT = KernelParams<VPT_C, NUM_EXPERTS_C, THREADS_PER_ROW_C,           \
                                 ROWS_PER_WARP_C, ROWS_PER_CTA_C, N_WARP_C>;        \
    moe_fused_gate_sqrtsoftplus_kernel_group1<VPT_C, NUM_EXPERTS_C, TOPK_C,         \
                                              ParamsT>                              \
        <<<grid, block_dim, 0, stream>>>(                                           \
            input.data_ptr<float>(), bias.data_ptr<float>(),                        \
            output.data_ptr<float>(), indices.data_ptr<int32_t>(), num_rows,        \
            num_fused_shared_experts, renormalize,                                  \
            static_cast<float>(routed_scaling_factor),                              \
            apply_routed_scaling_factor_on_output);                                 \
    dispatched = true;                                                              \
  } while (0)

#define DISPATCH_TOPK(TOPK_VAL)                                                     \
  case TOPK_VAL:                                                                    \
    if (num_experts == 256 && num_rows < 512) {                                     \
      LAUNCH_SQRTSOFTPLUS_GATE(TOPK_VAL, 256, 64, 4, 1);                            \
    } else if (num_experts == 256) {                                                \
      LAUNCH_SQRTSOFTPLUS_GATE(TOPK_VAL, 256, 32, 8, 2);                            \
    } else if (num_rows < 512) {                                                    \
      /* Match the tuned 384-expert group-1 layout in moe_fused_gate.cu:            \
       * one token per wave, eight contiguous FP32 values per active lane. */       \
      LAUNCH_SQRTSOFTPLUS_GATE(TOPK_VAL, 384, 64, 8, 1);                            \
    } else {                                                                        \
      LAUNCH_SQRTSOFTPLUS_GATE(TOPK_VAL, 384, 64, 8, 2);                            \
    }                                                                               \
    break

std::vector<torch::Tensor> moe_fused_gate_sqrtsoftplus(
    torch::Tensor& input,
    torch::Tensor& bias,
    int64_t topk,
    int64_t num_fused_shared_experts,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
  TORCH_CHECK(input.is_cuda(), "input must be on CUDA/ROCm device");
  TORCH_CHECK(bias.is_cuda(), "bias must be on CUDA/ROCm device");
  TORCH_CHECK(input.device() == bias.device(),
              "input and bias must be on the same device");
  TORCH_CHECK(input.scalar_type() == at::kFloat,
              "moe_fused_gate_sqrtsoftplus only supports float32 input");
  TORCH_CHECK(bias.scalar_type() == at::kFloat,
              "moe_fused_gate_sqrtsoftplus only supports float32 bias");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
  TORCH_CHECK(input.size(1) == bias.size(0),
              "input and bias must have the same num_experts");
  const int64_t num_experts = input.size(1);
  TORCH_CHECK(num_experts == 256 || num_experts == 384,
              "moe_fused_gate_sqrtsoftplus supports 256 or 384 experts, got ",
              num_experts);
  TORCH_CHECK(topk > 0 && topk <= MAX_TOPK,
              "topk must be in [1, ", MAX_TOPK, "], got ", topk);
  TORCH_CHECK(num_fused_shared_experts >= 0 &&
                  num_fused_shared_experts < topk,
              "num_fused_shared_experts must be >= 0 and < topk");
  TORCH_CHECK(routed_scaling_factor > 0.0,
              "routed_scaling_factor must be positive");

  const int64_t num_rows = input.size(0);
  auto output = torch::empty({num_rows, topk},
                             torch::TensorOptions()
                                 .dtype(torch::kFloat32)
                                 .device(input.device()));
  auto indices = torch::empty({num_rows, topk},
                              torch::TensorOptions()
                                  .dtype(torch::kInt32)
                                  .device(input.device()));
  if (num_rows == 0) return {output, indices};

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bool dispatched = false;
  switch (topk) {
    DISPATCH_TOPK(1);
    DISPATCH_TOPK(2);
    DISPATCH_TOPK(3);
    DISPATCH_TOPK(4);
    DISPATCH_TOPK(5);
    DISPATCH_TOPK(6);
    DISPATCH_TOPK(7);
    DISPATCH_TOPK(8);
    DISPATCH_TOPK(9);
    DISPATCH_TOPK(10);
    DISPATCH_TOPK(11);
    DISPATCH_TOPK(12);
    DISPATCH_TOPK(13);
    DISPATCH_TOPK(14);
    DISPATCH_TOPK(15);
    DISPATCH_TOPK(16);
    default:
      break;
  }
  TORCH_CHECK(dispatched, "failed to dispatch moe_fused_gate_sqrtsoftplus");
  return {output, indices};
}

#undef DISPATCH_TOPK
#undef LAUNCH_SQRTSOFTPLUS_GATE
#undef LIGHTOP_SHFL_XOR_WIDTH
#undef LIGHTOP_SHFL_BCAST_WIDTH

}  // namespace native
}  // namespace at
