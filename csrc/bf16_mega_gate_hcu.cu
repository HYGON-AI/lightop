// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <mutex>
#include <unordered_map>

#include "moe_gate_utils.cuh"

namespace at {
namespace native {
namespace {

// Four waves produce FP32 logits with the baseline K reduction order.
// The last producer CTA uses all eight waves, two tokens per wave, for Top-K.
template <int EXPERT_CAPACITY, int TOP_K, int HIDDEN = 0, bool PADDED = true>
__global__ __launch_bounds__(512) void bf16_mega_gate_kernel(
    const BFloat16* __restrict__ x, const BFloat16* __restrict__ w,
    const float* __restrict__ bias, const bool* __restrict__ mask,
    float* __restrict__ logits, float* __restrict__ routes,
    int* __restrict__ counters, float* __restrict__ weights,
    int64_t* __restrict__ ids, int m, int num_experts, int hidden_size,
    bool renormalize, float scale, bool apply_scale) {
  constexpr int BN = 32, BM = 16, NT = BN / 16;
  const int experts = PADDED ? num_experts : EXPERT_CAPACITY;
  const int hidden = HIDDEN ? HIDDEN : hidden_size;
  const int producer_ctas = (experts + BN - 1) / BN;
  const int wave = threadIdx.x / 64;
  const int lane = threadIdx.x % 64;
  const int row = lane % 16;
  const int col_group = lane / 16;
  const int ar = blockIdx.y * BM + row;
  const bool valid_row = row < BM && ar < m;
  float4_t acc[NT];
#pragma unroll
  for (int tile = 0; tile < NT; ++tile) acc[tile] = {0, 0, 0, 0};
  if (wave < 4) {
    for (int base = wave * 32; base < hidden; base += 128) {
      const int offset = base + col_group * 8;
      half4x2 av = {};
      if (valid_row)
        av = *reinterpret_cast<const half4x2*>(x + ar * hidden + offset);
#pragma unroll
      for (int tile = 0; tile < NT; ++tile) {
        const int br = blockIdx.x * BN + tile * 16 + row;
        half4x2 bv = {};
        if (!PADDED || br < experts)
          bv = *reinterpret_cast<const half4x2*>(w + br * hidden + offset);
        // Swapping operands makes each fragment span four tokens, so the
        // gate can load consecutive expert scores without a transpose pass.
        builtin_amdgcn_mmac<false>(bv.data[0], av.data[0], acc[tile]);
        builtin_amdgcn_mmac<false>(bv.data[1], av.data[1], acc[tile]);
      }
    }
  }
  __shared__ float4_t partial[2 * 64];
  __shared__ int is_last;
#pragma unroll
  for (int tile = 0; tile < NT; ++tile) {
#pragma unroll
    for (int active = 4; active > 1; active /= 2) {
      const int half = active / 2;
      if (wave >= half && wave < active)
        partial[(wave - half) * 64 + lane] = acc[tile];
      __syncthreads();
      if (wave < half) {
        const float4_t other = partial[wave * 64 + lane];
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[tile][j] += other[j];
      }
      __syncthreads();
    }
    if (wave == 0) {
      float scores[4];
      // Stage independent activations to overlap their instruction latency.
      float ex[4], sp[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) ex[j] = exp_func(acc[tile][j]);
#pragma unroll
      for (int j = 0; j < 4; ++j) sp[j] = log1pf(ex[j]);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        scores[j] = sqrtf(acc[tile][j] > 20.0f ? acc[tile][j] : sp[j]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int token = col_group + j * 4;
        const int expert = blockIdx.x * BN + tile * 16 + row;
        const int global_row = blockIdx.y * BM + token;
        if (token < BM && global_row < m && (!PADDED || expert < experts)) {
          const int index = global_row * experts + expert;
          logits[index] = acc[tile][j];
          routes[index] = scores[j];
        }
      }
    }
  }

  // Every writer publishes its global stores before the CTA leader announces
  // completion. A barrier alone is insufficient for cross-CTA visibility.
  if (wave == 0) __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int old = __hip_atomic_fetch_add(
        counters + blockIdx.y, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    is_last = old == producer_ctas - 1;
  }
  __syncthreads();
  if (!is_last) return;

  // Each consumer acquires the producers' release sequence, rather than
  // assuming the leader's acquire also invalidates another wave's cache.
  (void)__hip_atomic_load(counters + blockIdx.y, __ATOMIC_ACQUIRE,
                          __HIP_MEMORY_SCOPE_AGENT);
  const int token = blockIdx.y * BM + threadIdx.x / 32;
  moe_gate::topk_row<EXPERT_CAPACITY, TOP_K, PADDED>(
      routes, bias, logits, mask, weights, ids, token, m, experts, renormalize,
      scale, apply_scale);
  // All consumers finish before the counter is reset. The next launch on this
  // stream starts after this entire kernel; concurrent users need distinct
  // workspace (including independent/concurrently replayed graphs).
  __syncthreads();
  if (threadIdx.x == 0)
    __hip_atomic_store(counters + blockIdx.y, 0, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_AGENT);
}

// Align only the template capacity. Scratch keeps the real [M,E] stride,
// and the producer grid covers ceil(E/BN) tiles, including a guarded tail.
template <int CAPACITY, int TOP_K>
void launch_router(const Tensor& x, const Tensor& w, const Tensor& bias,
                   const Tensor& mask, Tensor logits, Tensor routes,
                   Tensor counters, Tensor weights, Tensor ids,
                   bool renormalize, float scale, bool apply_scale) {
  const int m = x.size(0), experts = w.size(0), hidden = x.size(1);
  const dim3 grid((experts + 31) / 32, (m + 15) / 16);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
#define LAUNCH_ROUTER(H, PAD)                                               \
  bf16_mega_gate_kernel<CAPACITY, TOP_K, H, PAD><<<grid, 512, 0, stream>>>( \
      x.data_ptr<BFloat16>(), w.data_ptr<BFloat16>(),                       \
      bias.defined() ? bias.data_ptr<float>() : nullptr,                    \
      mask.defined() ? mask.data_ptr<bool>() : nullptr,                     \
      logits.data_ptr<float>(), routes.data_ptr<float>(),                   \
      counters.data_ptr<int>(), weights.data_ptr<float>(),                  \
      ids.data_ptr<int64_t>(), m, experts, hidden, renormalize, scale,      \
      apply_scale)
  if constexpr (CAPACITY == 384 && TOP_K == 6) {
    if (experts == CAPACITY && hidden == 5120) {
      LAUNCH_ROUTER(5120, false);
      return;
    }
  }
  LAUNCH_ROUTER(0, true);
#undef LAUNCH_ROUTER
}

template <int CAPACITY>
void dispatch_topk(const Tensor& x, const Tensor& w, const Tensor& bias,
                   const Tensor& mask, Tensor logits, Tensor routes,
                   Tensor counters, Tensor weights, Tensor ids, int topk,
                   bool renormalize, float scale, bool apply_scale) {
#define TOPK_CASE(K)                                                           \
  case K:                                                                      \
    launch_router<CAPACITY, K>(x, w, bias, mask, logits, routes, counters,     \
                               weights, ids, renormalize, scale, apply_scale); \
    break
  switch (topk) {
    TOPK_CASE(1);
    TOPK_CASE(2);
    TOPK_CASE(3);
    TOPK_CASE(4);
    TOPK_CASE(5);
    TOPK_CASE(6);
    TOPK_CASE(7);
    TOPK_CASE(8);
    TOPK_CASE(9);
    TOPK_CASE(10);
    TOPK_CASE(11);
    TOPK_CASE(12);
    TOPK_CASE(13);
    TOPK_CASE(14);
    TOPK_CASE(15);
    TOPK_CASE(16);
    TOPK_CASE(17);
    TOPK_CASE(18);
    TOPK_CASE(19);
    TOPK_CASE(20);
    TOPK_CASE(21);
    TOPK_CASE(22);
    TOPK_CASE(23);
    TOPK_CASE(24);
    TOPK_CASE(25);
    TOPK_CASE(26);
    TOPK_CASE(27);
    TOPK_CASE(28);
    TOPK_CASE(29);
    TOPK_CASE(30);
    TOPK_CASE(31);
    TOPK_CASE(32);
  }
#undef TOPK_CASE
}

void check_tensor(const Tensor& t, const Tensor& x, ScalarType dtype,
                  const char* name) {
  TORCH_CHECK(
      t.device() == x.device() && t.scalar_type() == dtype && t.is_contiguous(),
      name, " has incorrect device, dtype or layout");
}
void check_router_inputs(const Tensor& x, const Tensor& weight,
                         const Tensor& bias, const Tensor& mask, int64_t topk,
                         double routed_scaling_factor) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() &&
                  x.scalar_type() == kBFloat16,
              "x must be contiguous GPU BF16 [M,H]");
  const auto* props = at::cuda::getDeviceProperties(x.get_device());
  const std::string arch = std::string(props->gcnArchName).substr(0, 6);
  TORCH_CHECK(arch == "gfx92a" || arch == "gfx928" || arch == "gfx936" || arch == "gfx938",
              "bf16_mega_gate_hcu requires gfx92a/gfx928/gfx936/gfx938");
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(k > 0 && k % 256 == 0, "H must be positive and divisible by 256");
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == k, "weight must be [E,H]");
  const int64_t experts = weight.size(0);
  TORCH_CHECK(experts > 0 && experts <= 512 && experts % 4 == 0,
              "E must be in [4,512] and divisible by 4");
  TORCH_CHECK(topk > 0 && topk <= 32 && topk <= experts,
              "topk must be in [1,min(E,32)]");
  TORCH_CHECK(
      k <= INT_MAX / experts && m <= INT_MAX / k && m <= INT_MAX / experts,
      "tensor dimensions exceed INT32 indexing");
  check_tensor(weight, x, kBFloat16, "weight");
  if (bias.defined()) {
    check_tensor(bias, x, kFloat, "bias");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == experts, "bias must be [E]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(bias.data_ptr()) % 32 == 0,
                "bias must be 32-byte aligned");
  }
  if (mask.defined()) {
    check_tensor(mask, x, kBool, "mask");
    TORCH_CHECK(mask.dim() == 1 && mask.size(0) == m, "mask must be [M]");
  }
  TORCH_CHECK(std::isfinite(routed_scaling_factor),
              "routed_scaling_factor must be finite");
  for (const auto& t : {x, weight})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "BF16 pointers must be 16-byte aligned");
}

Tensor get_stream_counters(const Tensor& x) {
  // H >= 256 and M * H <= INT_MAX bound every supported launch. Reserve the
  // full range once (2 MiB/stream) so later shapes cannot replace addresses
  // already referenced by captured graphs.
  constexpr int64_t kMaxTokenBlocks = (INT_MAX / 256 + 15) / 16;
  static std::mutex mutex;
  static std::unordered_map<c10::cuda::CUDAStream, Tensor> counters_by_stream;
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  std::lock_guard<std::mutex> lock(mutex);
  auto& counters = counters_by_stream[stream];
  if (!counters.defined()) {
    hipStreamCaptureStatus status;
    C10_CUDA_CHECK(hipStreamIsCapturing(stream, &status));
    TORCH_CHECK(
        status == hipStreamCaptureStatusNone,
        "warm up bf16_mega_gate_hcu on this stream before graph capture");
    counters = torch::zeros({kMaxTokenBlocks}, x.options().dtype(kInt));
  }
  // Calls/graphs sharing this capture stream must execute sequentially.
  return counters;
}

void run_router_out(const Tensor& x, const Tensor& weight, const Tensor& bias,
                    const Tensor& mask, Tensor logits, Tensor routes,
                    Tensor counters, Tensor weights, Tensor ids, int64_t topk,
                    bool renormalize, double routed_scaling_factor,
                    bool apply_scale) {
  const int64_t m = x.size(0), experts = weight.size(0), blocks = (m + 15) / 16;
  check_tensor(logits, x, kFloat, "logits");
  check_tensor(routes, x, kFloat, "routes");
  TORCH_CHECK(routes.sizes() == logits.sizes(), "routes must match logits");
  check_tensor(counters, x, kInt, "counters");
  check_tensor(weights, x, kFloat, "weights");
  check_tensor(ids, x, kLong, "ids");
  TORCH_CHECK(
      logits.dim() == 2 && logits.size(0) >= m && logits.size(1) == experts,
      "logits workspace is too small");
  TORCH_CHECK(counters.dim() == 1 && counters.numel() >= blocks,
              "counter workspace is too small");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == m &&
                  weights.size(1) == topk && ids.sizes() == weights.sizes(),
              "out must be FP32/INT64 [M,topk]");
  for (const auto& t : {logits, routes})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 32 == 0,
                "FP32 inputs must be 32-byte aligned");
  const std::vector<Tensor> outputs{logits, routes, counters, weights, ids};
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (const auto& input : {x, weight, bias, mask})
      if (input.defined()) assert_no_overlap(outputs[i], input);
    for (size_t j = i + 1; j < outputs.size(); ++j)
      assert_no_overlap(outputs[i], outputs[j]);
  }
  if (m == 0) return;
#define EXPERT_CASE(CAPACITY)                                                \
  case CAPACITY:                                                             \
    dispatch_topk<CAPACITY>(x, weight, bias, mask, logits, routes, counters, \
                            weights, ids, topk, renormalize,                 \
                            routed_scaling_factor, apply_scale);             \
    break
  switch ((experts + 127) / 128 * 128) {
    EXPERT_CASE(128);
    EXPERT_CASE(256);
    EXPERT_CASE(384);
    EXPERT_CASE(512);
  }
#undef EXPERT_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

std::vector<Tensor> bf16_mega_gate_hcu_impl(
    const Tensor& x, const Tensor& weight,
    const torch::optional<Tensor>& optional_bias, int64_t topk,
    bool renormalize, double routed_scaling_factor, bool apply_scale,
    const torch::optional<std::vector<Tensor>>& out,
    const torch::optional<Tensor>& optional_mask) {
  const Tensor bias = optional_bias.value_or(Tensor());
  const Tensor mask = optional_mask.value_or(Tensor());
  check_router_inputs(x, weight, bias, mask, topk, routed_scaling_factor);
  const c10::cuda::CUDAGuard device_guard(x.device());
  const auto m = x.size(0), experts = weight.size(0);
  TORCH_CHECK(!out.has_value() || out->size() == 2,
              "out must be (weights_fp32, ids_int64)");
  Tensor weights = out.has_value()
                       ? (*out)[0]
                       : torch::empty({m, topk}, x.options().dtype(kFloat));
  Tensor ids = out.has_value()
                   ? (*out)[1]
                   : torch::empty({m, topk}, x.options().dtype(kLong));
  // PyTorch's allocator manages scratch lifetime, including graph capture.
  auto logits = torch::empty({m, experts}, x.options().dtype(kFloat));
  auto routes = torch::empty_like(logits);
  auto counters = m == 0 ? torch::empty({0}, x.options().dtype(kInt))
                         : get_stream_counters(x);
  run_router_out(x, weight, bias, mask, logits, routes, counters, weights, ids,
                 topk, renormalize, routed_scaling_factor, apply_scale);
  return {weights, ids, logits};
}

std::tuple<Tensor, Tensor> bf16_mega_gate_hcu(
    const Tensor& x, const Tensor& weight, int64_t num_topk,
    bool use_shared_as_routed, int64_t num_shared_experts,
    double routed_scaling_factor, int64_t ep_rank,
    const std::string& scoring_func, const torch::optional<Tensor>& mask,
    const torch::optional<Tensor>& bias,
    const torch::optional<Tensor>& image_bias,
    const torch::optional<Tensor>& image_token_mask,
    const torch::optional<Tensor>& fix_routing_mask,
    const torch::optional<Tensor>& to_physical_map,
    const torch::optional<Tensor>& logical_count,
    const torch::optional<Tensor>& unmapped_topk_idx,
    const torch::optional<Tensor>& force_random,
    const torch::optional<std::tuple<Tensor, Tensor>>& out) {
  TORCH_CHECK(scoring_func == "sqrtsoftplus",
              "scoring_func must be sqrtsoftplus");
  TORCH_CHECK(ep_rank >= 0, "ep_rank must be nonnegative");
  TORCH_CHECK_NOT_IMPLEMENTED(
      !use_shared_as_routed,
      "HCU mega gate does not support shared-as-routed experts");
  TORCH_CHECK_NOT_IMPLEMENTED(!image_bias && !image_token_mask,
                              "HCU mega gate does not support image routing");
  TORCH_CHECK_NOT_IMPLEMENTED(
      !fix_routing_mask && !unmapped_topk_idx,
      "HCU mega gate does not support fixed/unmapped routing");
  TORCH_CHECK_NOT_IMPLEMENTED(
      !to_physical_map && !logical_count,
      "HCU mega gate does not support physical expert mapping");
  TORCH_CHECK_NOT_IMPLEMENTED(!force_random,
                              "HCU mega gate does not support random routing");
  // Like DeepGEMM, num_shared_experts is unused when use_shared_as_routed is
  // false.
  (void)num_shared_experts;
  torch::optional<std::vector<Tensor>> buffers;
  if (out) buffers = std::vector<Tensor>{std::get<1>(*out), std::get<0>(*out)};
  auto result =
      bf16_mega_gate_hcu_impl(x, weight, bias, num_topk, true,
                              routed_scaling_factor, true, buffers, mask);
  return {result[1], result[0]};
}

}  // namespace native
}  // namespace at
