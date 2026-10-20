#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <tuple>

namespace lightop::dsv41 {
namespace {
constexpr int kThreads = 256;

template <typename req_t, typename pos_t>
__global__ void pair_read_kernel(
    const float* kv, const float* score, const req_t* req,
    const pos_t* pos, const bool* pad, const float* state,
    float* partner_kv, float* partner_score, int64_t dim,
    int64_t kv_stride, int64_t score_stride, int64_t state_stride,
    int64_t sentinel, int64_t ring) {
  const int64_t row = blockIdx.x;
  const bool adjacent = row > 0 && !pad[row] && !pad[row - 1] &&
      req[row] == req[row - 1] &&
      static_cast<int64_t>(pos[row]) == static_cast<int64_t>(pos[row - 1]) + 1;
  const int64_t previous = static_cast<int64_t>(pos[row]) - 1;
  const int64_t slot = pad[row] || previous < 0
      ? sentinel : static_cast<int64_t>(req[row]) * ring + previous % ring;
  for (int64_t col = threadIdx.x; col < dim; col += blockDim.x) {
    partner_kv[row * dim + col] = adjacent
        ? kv[(row - 1) * kv_stride + col] : state[slot * state_stride + col];
    partner_score[row * dim + col] = adjacent
        ? score[(row - 1) * score_stride + col]
        : state[slot * state_stride + dim + col];
  }
}

template <typename req_t, typename pos_t>
__global__ void pair_write_kernel(
    const float* kv, const float* score, const req_t* req,
    const pos_t* pos, const bool* pad, float* state,
    int64_t tokens, int64_t dim, int64_t kv_stride, int64_t score_stride,
    int64_t state_stride, int64_t sentinel, int64_t ring) {
  const int64_t row = blockIdx.x;
  if (row == tokens) {
    for (int64_t col = threadIdx.x; col < dim; col += blockDim.x) {
      state[sentinel * state_stride + col] = 0.0f;
      state[sentinel * state_stride + dim + col] = -INFINITY;
    }
    return;
  }
  if (pad[row] || static_cast<int64_t>(pos[row]) < 0) return;
  if (row + ring < tokens && req[row] == req[row + ring] && !pad[row + ring]) return;
  const int64_t slot = static_cast<int64_t>(req[row]) * ring +
      static_cast<int64_t>(pos[row]) % ring;
  for (int64_t col = threadIdx.x; col < dim; col += blockDim.x) {
    state[slot * state_stride + col] = kv[row * kv_stride + col];
    state[slot * state_stride + dim + col] = score[row * score_stride + col];
  }
}

template <typename req_t, typename pos_t>
void launch_pair(
    const torch::Tensor& kv, const torch::Tensor& score,
    const torch::Tensor& req, const torch::Tensor& pos, const torch::Tensor& pad,
    torch::Tensor& state, int64_t ring, torch::Tensor& out_kv, torch::Tensor& out_score) {
  const auto stream = at::hip::getCurrentHIPStream();
  const auto tokens = kv.size(0), dim = kv.size(1), sentinel = state.size(0) - 1;
  pair_read_kernel<req_t, pos_t><<<tokens, kThreads, 0, stream>>>(
      kv.data_ptr<float>(), score.data_ptr<float>(), req.data_ptr<req_t>(),
      pos.data_ptr<pos_t>(), pad.data_ptr<bool>(), state.data_ptr<float>(),
      out_kv.data_ptr<float>(), out_score.data_ptr<float>(), dim,
      kv.stride(0), score.stride(0), state.stride(0), sentinel, ring);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "dsv41_pair_partners: read launch failed");
  // The stream boundary is a global read-before-write ordering point. A block
  // barrier in one combined grid cannot protect carry reads from other CTAs.
  pair_write_kernel<req_t, pos_t><<<tokens + 1, kThreads, 0, stream>>>(
      kv.data_ptr<float>(), score.data_ptr<float>(), req.data_ptr<req_t>(),
      pos.data_ptr<pos_t>(), pad.data_ptr<bool>(), state.data_ptr<float>(),
      tokens, dim, kv.stride(0), score.stride(0), state.stride(0), sentinel, ring);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "dsv41_pair_partners: write launch failed");
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor> pair_partners(
    const torch::Tensor& kv, const torch::Tensor& score,
    const torch::Tensor& req, const torch::Tensor& pos, const torch::Tensor& pad,
    torch::Tensor& state, int64_t ring) {
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == at::kFloat && kv.dim() == 2 &&
              kv.size(1) > 0 && kv.stride(1) == 1 && kv.stride(0) >= kv.size(1),
              "dsv41_pair_partners: kv must be FP32 [T,D] with inner stride 1");
  TORCH_CHECK(score.device() == kv.device() && score.scalar_type() == at::kFloat &&
              score.sizes() == kv.sizes() && score.stride(1) == 1 && score.stride(0) >= kv.size(1),
              "dsv41_pair_partners: score must match kv shape/device/dtype");
  TORCH_CHECK(state.device() == kv.device() && state.scalar_type() == at::kFloat &&
              state.dim() == 2 && state.size(0) > 0 && state.size(1) == 2 * kv.size(1) &&
              state.is_contiguous(), "dsv41_pair_partners: state must be contiguous FP32 [S,2D]");
  TORCH_CHECK(ring > 0 && state.size(0) > ring,
              "dsv41_pair_partners: invalid ring or state capacity");
  for (const auto& index : {req, pos}) {
    TORCH_CHECK(index.device() == kv.device() && index.dim() == 1 &&
                index.size(0) == kv.size(0) && index.is_contiguous() &&
                (index.scalar_type() == at::kInt || index.scalar_type() == at::kLong),
                "dsv41_pair_partners: req and pos must be contiguous int32/int64 [T]");
  }
  TORCH_CHECK(pad.device() == kv.device() && pad.scalar_type() == at::kBool &&
              pad.dim() == 1 && pad.size(0) == kv.size(0) && pad.is_contiguous(),
              "dsv41_pair_partners: pad must be contiguous bool [T]");
  if (kv.size(0) == 0) return {kv, score};
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(kv.device());
  auto out_kv = torch::empty(kv.sizes(), kv.options());
  auto out_score = torch::empty(kv.sizes(), kv.options());
  if (req.scalar_type() == at::kInt && pos.scalar_type() == at::kInt) {
    launch_pair<int32_t, int32_t>(kv, score, req, pos, pad, state, ring, out_kv, out_score);
  } else if (req.scalar_type() == at::kInt) {
    launch_pair<int32_t, int64_t>(kv, score, req, pos, pad, state, ring, out_kv, out_score);
  } else if (pos.scalar_type() == at::kInt) {
    launch_pair<int64_t, int32_t>(kv, score, req, pos, pad, state, ring, out_kv, out_score);
  } else {
    launch_pair<int64_t, int64_t>(kv, score, req, pos, pad, state, ring, out_kv, out_score);
  }
  return {out_kv, out_score};
}
}  // namespace lightop::dsv41
