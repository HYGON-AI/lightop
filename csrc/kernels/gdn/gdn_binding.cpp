// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// PyTorch bindings for the GDN recurrent HIP kernel (gdn_recurrent.hip).
//
// gdn_recurrent.h is the single source of truth for the entry points; this
// file only adds the argument marshalling, so callers pass tensors and
// semantic flags instead of the C ABI. Token strides, hratio, scale and the
// softplus literals are derived here, mirroring the adapter in
// python/lightop/gdn.py.
//
// A_log and dt_bias are read as const float* by the kernel: bfloat16 inputs
// must be cast first (lightop.gdn.f32) or every head's decay is wrong.

// This file deliberately avoids torch's c10/cuda host API: the DTK torch
// headers are CUDA-flavored (cuda_runtime_api.h), so a host TU that includes
// them references cuda* symbols the HIP torch libraries do not export. The
// stream handle comes from lightop.gdn's _stream() and device selection uses
// the HIP runtime directly.
#include <torch/extension.h>

#include <hip/hip_runtime.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

extern "C" {
#include "kernels/gdn/gdn_recurrent.h"
}

namespace lightop::gdn {
namespace {

// softplus(beta * x + threshold) literals the recurrent kernel expects.
constexpr float kSoftplusBeta = 1.0f;
constexpr float kSoftplusThreshold = 20.0f;
// Conv kernel contract: width 4, at most 16 tokens per call, state_len >= 3.
constexpr int64_t kConvWidth = 4;
constexpr int64_t kConvMaxSeqlen = 16;

// Opaque handle the C ABI takes: torch's current stream for the tensor's
// device, fetched by the Python wrapper.
void* stream_handle(int64_t stream) {
  return reinterpret_cast<void*>(static_cast<uintptr_t>(stream));
}

class DeviceGuard {
 public:
  explicit DeviceGuard(int device) : device_(device) {
    if (hipGetDevice(&saved_) != hipSuccess) {
      saved_ = -1;
    }
    if (saved_ != device_) {
      hipSetDevice(device_);
    }
  }
  ~DeviceGuard() {
    if (saved_ >= 0 && saved_ != device_) {
      hipSetDevice(saved_);
    }
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int device_;
  int saved_ = -1;
};

void check_launch() {
  const hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "GDN HIP launch failed: ",
              hipGetErrorString(err));
}

// Stride between tokens: q/k/v are (1, seq, H, D), a/b are (seq, HV).
int64_t tok_stride(const at::Tensor& x) {
  return x.dim() >= 3 ? x.stride(1) : x.stride(0);
}

const void* data_or_null(const std::optional<at::Tensor>& x) {
  return x.has_value() ? x->data_ptr() : nullptr;
}

void expect_bf16(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name,
              " must be bfloat16, got ", t.scalar_type());
}

void expect_f32(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.scalar_type() == at::kFloat, name,
              " must be float32 (cast A_log/dt_bias with lightop.gdn.f32), got ",
              t.scalar_type());
}

void expect_i32(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32, got ",
              t.scalar_type());
}

void expect_contiguous(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void expect_cuda(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
}

void expect_same_device(const at::Tensor& ref, const at::Tensor& t,
                        const char* name) {
  TORCH_CHECK(ref.device() == t.device(), name, " must be on ", ref.device(),
              ", got ", t.device());
}

// float(1.0 / sqrt(K)) as the plugin computes it (double divide, float cast).
float rsqrt_scale(int64_t kdim) {
  return static_cast<float>(1.0 / std::sqrt(static_cast<double>(kdim)));
}

void check_head_layout(int64_t h, int64_t hv) {
  TORCH_CHECK(h > 0 && hv > 0 && hv % h == 0,
              "HV must be a positive multiple of H, got H=", h, " HV=", hv);
}

// The recurrent kernel indexes k[K-1] and grids V in BV-wide blocks; anything
// else reads past the head or silently drops the tail.
void check_kv_dims(int64_t kdim, int64_t vdim) {
  TORCH_CHECK(kdim == 128, "GDN requires K=128, got ", kdim);
  TORCH_CHECK(vdim % 32 == 0, "GDN requires V % 32 == 0, got ", vdim);
}

// Standalone norm pass: q/k l2norm (dense fp32) plus eg/beta for each hv.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> gdn_norm_pass(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& a,
    const at::Tensor& b, const at::Tensor& A_log, const at::Tensor& dt_bias,
    const at::Tensor& qn, const at::Tensor& kn, const at::Tensor& eg,
    const at::Tensor& beta, int64_t stream, std::optional<int64_t> hv) {
  expect_bf16(q, "q");
  expect_bf16(k, "k");
  expect_bf16(a, "a");
  expect_bf16(b, "b");
  expect_f32(A_log, "A_log");
  expect_f32(dt_bias, "dt_bias");
  expect_contiguous(A_log, "A_log");
  expect_contiguous(dt_bias, "dt_bias");
  expect_f32(qn, "qn");
  expect_f32(kn, "kn");
  expect_f32(eg, "eg");
  expect_f32(beta, "beta");
  expect_contiguous(qn, "qn");
  expect_contiguous(kn, "kn");
  expect_contiguous(eg, "eg");
  expect_contiguous(beta, "beta");
  expect_cuda(q, "q");
  expect_same_device(q, a, "a");
  expect_same_device(q, qn, "qn");
  expect_same_device(q, eg, "eg");

  const int64_t h = q.size(-2);
  const int64_t kdim = q.size(-1);
  TORCH_CHECK(kdim == 128, "GDN norm pass requires K=128, got ", kdim);
  const int64_t hv_dim = hv.value_or(a.size(-1));
  check_head_layout(h, hv_dim);
  const int64_t hratio = hv_dim / h;
  const int64_t tokens = q.numel() / (h * kdim);
  TORCH_CHECK(tokens > 0, "q must hold at least one token");
  TORCH_CHECK(qn.numel() >= tokens * h * kdim,
              "qn is too small for ", tokens, " tokens");
  TORCH_CHECK(kn.numel() >= tokens * h * kdim,
              "kn is too small for ", tokens, " tokens");
  TORCH_CHECK(eg.numel() >= tokens * hv_dim, "eg is too small for ", tokens,
              " tokens");
  TORCH_CHECK(beta.numel() >= tokens * hv_dim, "beta is too small for ", tokens,
              " tokens");

  DeviceGuard guard(q.device().index());
  gdn_verify_norm_pass(
      q.data_ptr(), k.data_ptr(), a.data_ptr(), b.data_ptr(), A_log.data_ptr(),
      dt_bias.data_ptr(), qn.data_ptr(), kn.data_ptr(), eg.data_ptr(),
      beta.data_ptr(), static_cast<int>(tokens * h), static_cast<int>(h),
      static_cast<int>(hv_dim), static_cast<int>(kdim),
      static_cast<int>(hratio), static_cast<int>(tok_stride(q)),
      static_cast<int>(tok_stride(k)), static_cast<int>(tok_stride(a)),
      static_cast<int>(tok_stride(b)), kSoftplusBeta, kSoftplusThreshold,
      rsqrt_scale(kdim), stream_handle(stream));
  check_launch();
  return {qn, kn, eg, beta};
}

// B+ verify: writes the snapshot row selected by store/store_row. With
// prenorm the q/k l2norm and (optionally) the eg/beta gating are hoisted into
// one launch of gdn_norm_pass before the recurrent kernel.
at::Tensor gdn_verify(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
    const at::Tensor& dt_bias, const std::optional<at::Tensor>& cu_seqlens,
    const at::Tensor& h0_source, const at::Tensor& h0_indices,
    const at::Tensor& snap_buf, const at::Tensor& snap_idx,
    int64_t cache_steps, int64_t draft_token_num, const at::Tensor& o,
    int64_t stream, const std::optional<at::Tensor>& prev_step,
    const std::optional<at::Tensor>& store,
    const std::optional<at::Tensor>& store_row, bool prenorm,
    bool prenorm_gate, const std::optional<at::Tensor>& qn,
    const std::optional<at::Tensor>& kn, const std::optional<at::Tensor>& eg,
    const std::optional<at::Tensor>& beta) {
  expect_bf16(q, "q");
  expect_bf16(k, "k");
  expect_bf16(v, "v");
  expect_bf16(a, "a");
  expect_bf16(b, "b");
  expect_bf16(o, "o");
  expect_f32(A_log, "A_log");
  expect_f32(dt_bias, "dt_bias");
  expect_f32(snap_buf, "snap_flat");
  expect_contiguous(A_log, "A_log");
  expect_contiguous(dt_bias, "dt_bias");
  expect_i32(h0_indices, "h0_indices");
  expect_i32(snap_idx, "snap_idx");
  expect_cuda(q, "q");
  expect_same_device(q, v, "v");
  expect_same_device(q, o, "o");
  expect_same_device(q, snap_idx, "snap_idx");
  if (cu_seqlens.has_value()) {
    expect_i32(*cu_seqlens, "cu_seqlens");
    expect_same_device(q, *cu_seqlens, "cu_seqlens");
  }

  const int64_t hv = v.size(-2);
  const int64_t h = q.size(-2);
  const int64_t kdim = k.size(-1);
  const int64_t vdim = v.size(-1);
  check_kv_dims(kdim, vdim);
  check_head_layout(h, hv);
  const int64_t hratio = hv / h;
  const int64_t n = snap_idx.size(0);

  // store/store_row are per batch: defaults mirror gdn_recurrent.py.verify().
  at::Tensor store_t = store.has_value()
                           ? *store
                           : at::full({n}, draft_token_num - 1,
                                      q.options().dtype(at::kInt));
  expect_i32(store_t, "store");
  expect_same_device(q, store_t, "store");
  at::Tensor row_t;
  if (store_row.has_value()) {
    expect_i32(*store_row, "store_row");
    expect_same_device(q, *store_row, "store_row");
    row_t = *store_row;
  } else if (prev_step.has_value()) {
    TORCH_CHECK(prev_step->scalar_type() == at::kInt ||
                    prev_step->scalar_type() == at::kLong,
                "prev_step must be int32 or int64, got ",
                prev_step->scalar_type());
    expect_same_device(q, *prev_step, "prev_step");
    auto safe = snap_idx.to(at::kLong).clamp(0, prev_step->numel() - 1);
    auto prev = prev_step->index_select(0, safe).to(at::kInt);
    row_t = at::remainder(prev + 1, cache_steps).to(at::kInt);
  } else {
    row_t = store_t;
  }

  DeviceGuard guard(q.device().index());
  const bool varlen = cu_seqlens.has_value();
  const int64_t stride_q = tok_stride(q);
  const int64_t stride_k = tok_stride(k);
  const int64_t stride_v = tok_stride(v);
  const int64_t stride_a = tok_stride(a);
  const int64_t stride_b = tok_stride(b);
  const float scale = rsqrt_scale(kdim);
  if (prenorm) {
    TORCH_CHECK(qn.has_value() && kn.has_value() && eg.has_value() &&
                    beta.has_value(),
                "prenorm verify requires qn/kn/eg/beta scratch buffers");
    const int64_t tokens = q.numel() / (h * kdim);
    gdn_verify_norm_pass(
        q.data_ptr(), k.data_ptr(), a.data_ptr(), b.data_ptr(),
        A_log.data_ptr(), dt_bias.data_ptr(), qn->data_ptr(), kn->data_ptr(),
        eg->data_ptr(), beta->data_ptr(), static_cast<int>(tokens * h),
        static_cast<int>(h), static_cast<int>(hv), static_cast<int>(kdim),
        static_cast<int>(hratio), static_cast<int>(stride_q),
        static_cast<int>(stride_k), static_cast<int>(stride_a),
        static_cast<int>(stride_b), kSoftplusBeta, kSoftplusThreshold, scale,
        stream_handle(stream));
    gdn_recurrent_update_bplus_prenorm(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), a.data_ptr(), b.data_ptr(),
        A_log.data_ptr(), dt_bias.data_ptr(), data_or_null(cu_seqlens),
        h0_source.data_ptr(), h0_indices.data_ptr(), o.data_ptr(),
        snap_buf.data_ptr(), snap_idx.data_ptr(), static_cast<int>(cache_steps),
        store_t.data_ptr(), row_t.data_ptr(), static_cast<int>(n),
        static_cast<int>(draft_token_num), static_cast<int>(hv),
        static_cast<int>(kdim), static_cast<int>(vdim),
        static_cast<int>(hratio), static_cast<int>(stride_q),
        static_cast<int>(stride_k), static_cast<int>(stride_v),
        static_cast<int>(stride_a), static_cast<int>(stride_b), scale,
        kSoftplusBeta, kSoftplusThreshold, 1, varlen ? 1 : 0, 1,
        qn->data_ptr(), kn->data_ptr(), eg->data_ptr(), beta->data_ptr(),
        static_cast<int>(h * kdim), static_cast<int>(hv),
        prenorm_gate ? 1 : 0, stream_handle(stream));
  } else {
    gdn_recurrent_update_bplus(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), a.data_ptr(), b.data_ptr(),
        A_log.data_ptr(), dt_bias.data_ptr(), data_or_null(cu_seqlens),
        h0_source.data_ptr(), h0_indices.data_ptr(), o.data_ptr(),
        snap_buf.data_ptr(), snap_idx.data_ptr(), static_cast<int>(cache_steps),
        store_t.data_ptr(), row_t.data_ptr(), static_cast<int>(n),
        static_cast<int>(draft_token_num), static_cast<int>(hv),
        static_cast<int>(kdim), static_cast<int>(vdim),
        static_cast<int>(hratio), static_cast<int>(stride_q),
        static_cast<int>(stride_k), static_cast<int>(stride_v),
        static_cast<int>(stride_a), static_cast<int>(stride_b), scale,
        kSoftplusBeta, kSoftplusThreshold, 1, varlen ? 1 : 0, 1,
        stream_handle(stream));
  }
  check_launch();
  return o;
}

// Full-snapshot entry: same kernel and inputs, every step written. A/B only.
at::Tensor gdn_verify_full(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
    const at::Tensor& dt_bias, const std::optional<at::Tensor>& cu_seqlens,
    const at::Tensor& h0_source, const at::Tensor& h0_indices,
    const at::Tensor& snap_buf, const at::Tensor& snap_idx,
    int64_t cache_steps, int64_t draft_token_num, const at::Tensor& o,
    int64_t stream) {
  expect_bf16(q, "q");
  expect_bf16(k, "k");
  expect_bf16(v, "v");
  expect_bf16(a, "a");
  expect_bf16(b, "b");
  expect_bf16(o, "o");
  expect_f32(A_log, "A_log");
  expect_f32(dt_bias, "dt_bias");
  expect_f32(snap_buf, "snap_flat");
  expect_contiguous(A_log, "A_log");
  expect_contiguous(dt_bias, "dt_bias");
  expect_i32(h0_indices, "h0_indices");
  expect_i32(snap_idx, "snap_idx");
  expect_cuda(q, "q");
  expect_same_device(q, o, "o");
  if (cu_seqlens.has_value()) {
    expect_i32(*cu_seqlens, "cu_seqlens");
    expect_same_device(q, *cu_seqlens, "cu_seqlens");
  }
  const int64_t hv = v.size(-2);
  const int64_t h = q.size(-2);
  const int64_t kdim = k.size(-1);
  const int64_t vdim = v.size(-1);
  check_kv_dims(kdim, vdim);
  check_head_layout(h, hv);
  const int64_t hratio = hv / h;
  const int64_t n = snap_idx.size(0);

  DeviceGuard guard(q.device().index());
  gdn_recurrent_update_nofma(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), a.data_ptr(), b.data_ptr(),
      A_log.data_ptr(), dt_bias.data_ptr(), data_or_null(cu_seqlens),
      h0_source.data_ptr(), h0_indices.data_ptr(), o.data_ptr(),
      snap_buf.data_ptr(), snap_idx.data_ptr(), static_cast<int>(cache_steps),
      static_cast<int>(n), static_cast<int>(draft_token_num),
      static_cast<int>(hv), static_cast<int>(kdim),
      static_cast<int>(v.size(-1)), static_cast<int>(hratio),
      static_cast<int>(tok_stride(q)), static_cast<int>(tok_stride(k)),
      static_cast<int>(tok_stride(v)), static_cast<int>(tok_stride(a)),
      static_cast<int>(tok_stride(b)), rsqrt_scale(kdim), kSoftplusBeta,
      kSoftplusThreshold, 1, cu_seqlens.has_value() ? 1 : 0, 1,
      stream_handle(stream));
  check_launch();
  return o;
}

// Replay state only (no o): slots with h0_rows < 0 exit immediately.
at::Tensor gdn_compensate(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
    const at::Tensor& dt_bias, const at::Tensor& h0_source,
    const at::Tensor& h0_rows, const at::Tensor& snap_buf,
    const at::Tensor& snap_idx, int64_t cache_steps,
    const at::Tensor& store_step, int64_t n_seq, int64_t tok_per_seq,
    const at::Tensor& o, int64_t stream) {
  expect_bf16(q, "q");
  expect_bf16(k, "k");
  expect_bf16(v, "v");
  expect_bf16(a, "a");
  expect_bf16(b, "b");
  expect_bf16(o, "o");
  expect_f32(A_log, "A_log");
  expect_f32(dt_bias, "dt_bias");
  expect_f32(snap_buf, "snap_flat");
  expect_contiguous(A_log, "A_log");
  expect_contiguous(dt_bias, "dt_bias");
  expect_i32(h0_rows, "h0_rows");
  expect_i32(snap_idx, "snap_idx");
  expect_i32(store_step, "store_step");
  expect_cuda(q, "q");
  expect_same_device(q, h0_source, "ssm_states");
  expect_same_device(q, store_step, "store_step");
  const int64_t hv = v.size(-2);
  const int64_t h = q.size(-2);
  const int64_t kdim = k.size(-1);
  const int64_t vdim = v.size(-1);
  check_kv_dims(kdim, vdim);
  check_head_layout(h, hv);

  DeviceGuard guard(q.device().index());
  gdn_recurrent_update_bplus_state(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), a.data_ptr(), b.data_ptr(),
      A_log.data_ptr(), dt_bias.data_ptr(), nullptr, h0_source.data_ptr(),
      h0_rows.data_ptr(), o.data_ptr(), snap_buf.data_ptr(),
      snap_idx.data_ptr(), static_cast<int>(cache_steps),
      store_step.data_ptr(), nullptr, static_cast<int>(n_seq),
      static_cast<int>(tok_per_seq), static_cast<int>(hv),
      static_cast<int>(kdim), static_cast<int>(v.size(-1)),
      static_cast<int>(hv / h), static_cast<int>(tok_stride(q)),
      static_cast<int>(tok_stride(k)), static_cast<int>(tok_stride(v)),
      static_cast<int>(tok_stride(a)), static_cast<int>(tok_stride(b)),
      rsqrt_scale(kdim), kSoftplusBeta, kSoftplusThreshold, 1, 0, 1,
      stream_handle(stream));
  check_launch();
  return o;
}

// One-launch state replay. layer_ptrs is an int64 [L, 15] device table with the
// columns the kernel reads (see GDN_MULTI_COLS in gdn_recurrent.hip):
// q, k, v, a, b, A_log, dt_bias, h0_source, o, snap_buf,
// stride_q, stride_k, stride_v, stride_a, stride_b.
at::Tensor gdn_compensate_multi(
    const at::Tensor& layer_ptrs, const at::Tensor& h0_indices,
    const at::Tensor& snap_idx, int64_t cache_steps,
    const at::Tensor& store_step, int64_t n_seq, int64_t tok_per_seq,
    int64_t hv, int64_t h, int64_t kdim, int64_t vdim, int64_t stream) {
  TORCH_CHECK(layer_ptrs.scalar_type() == at::kLong,
              "layer_ptrs must be int64, got ", layer_ptrs.scalar_type());
  TORCH_CHECK(layer_ptrs.dim() == 2 && layer_ptrs.size(1) == 15,
              "layer_ptrs must be [L, 15], got ", layer_ptrs.sizes());
  expect_contiguous(layer_ptrs, "layer_ptrs");
  expect_i32(h0_indices, "h0_indices");
  expect_i32(snap_idx, "snap_idx");
  expect_i32(store_step, "store_step");
  expect_cuda(layer_ptrs, "layer_ptrs");
  expect_same_device(layer_ptrs, snap_idx, "snap_idx");
  check_kv_dims(kdim, vdim);
  check_head_layout(h, hv);
  const int64_t layers = layer_ptrs.size(0);

  DeviceGuard guard(layer_ptrs.device().index());
  gdn_recurrent_update_bplus_state_multi(
      layer_ptrs.data_ptr(), h0_indices.data_ptr(), snap_idx.data_ptr(),
      static_cast<int>(cache_steps), store_step.data_ptr(),
      static_cast<int>(layers), static_cast<int>(n_seq),
      static_cast<int>(tok_per_seq), static_cast<int>(hv),
      static_cast<int>(kdim), static_cast<int>(vdim),
      static_cast<int>(hv / h), rsqrt_scale(kdim), kSoftplusBeta,
      kSoftplusThreshold, 1, 0, 1, stream_handle(stream));
  check_launch();
  return layer_ptrs;
}

// ---------------------------------------------------------------------------
// causal_conv1d update (target verify). Returns None when the call falls
// outside what the HIP kernel implements; the caller keeps the Triton kernel.
// ---------------------------------------------------------------------------

std::optional<std::string> conv_unsupported_reason_impl(
    const at::Tensor& x, const at::Tensor& conv_state, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& cache_seqlens,
    const std::optional<at::Tensor>& conv_state_indices,
    const std::optional<at::Tensor>& num_accept_tokens,
    const std::optional<at::Tensor>& window,
    const std::optional<at::Tensor>& window_indices,
    const std::optional<at::Tensor>& nxt,
    const std::optional<at::Tensor>& sib,
    const std::optional<at::Tensor>& parent,
    const std::optional<int64_t>& pad_slot_id) {
  const int64_t seqlen = x.dim() == 3 ? x.size(-1) : 1;
  if (x.dim() != 2 && x.dim() != 3) {
    return "x.dim=" + std::to_string(x.dim());
  }
  if (weight.dim() != 2 || weight.size(1) != kConvWidth) {
    return "weight=(" + std::to_string(weight.size(0)) + ", " +
           std::to_string(weight.dim() > 1 ? weight.size(1) : 0) + ")";
  }
  if (conv_state.dim() != 3) {
    return "conv_state.dim=" + std::to_string(conv_state.dim());
  }
  if (seqlen < 1 || seqlen > kConvMaxSeqlen) {
    return "seqlen=" + std::to_string(seqlen);
  }
  const int64_t dim = x.dim() >= 2 ? x.size(1) : 0;
  if (dim != weight.size(0) || dim != conv_state.size(1)) {
    return "dim mismatch x=" + std::to_string(dim) +
           " w=" + std::to_string(weight.size(0)) +
           " state=" + std::to_string(conv_state.size(1));
  }
  if (conv_state.size(2) < kConvWidth - 1) {
    return "state_len=" + std::to_string(conv_state.size(2));
  }
  if (x.scalar_type() != at::kBFloat16 ||
      weight.scalar_type() != at::kBFloat16) {
    return "dtype x=" + std::string(c10::toString(x.scalar_type())) +
           " w=" + std::string(c10::toString(weight.scalar_type()));
  }
  if (conv_state.scalar_type() != at::kBFloat16) {
    return "conv_state.dtype=" +
           std::string(c10::toString(conv_state.scalar_type()));
  }
  if (bias.has_value() &&
      (bias->scalar_type() != at::kBFloat16 || !bias->is_contiguous())) {
    return std::string("bias=") + c10::toString(bias->scalar_type()) +
           "/contig=" + (bias->is_contiguous() ? "true" : "false");
  }
  if (cache_seqlens.has_value() || num_accept_tokens.has_value()) {
    return "circular / multi-accept conv_state is not implemented";
  }
  if (conv_state_indices.has_value() &&
      (conv_state_indices->scalar_type() != at::kInt ||
       conv_state_indices->dim() != 1)) {
    return std::string("conv_state_indices=") +
           c10::toString(conv_state_indices->scalar_type());
  }
  if (window.has_value()) {
    if (window->scalar_type() != at::kBFloat16 || window->dim() != 4) {
      return std::string("window=") + c10::toString(window->scalar_type()) +
             "/" + std::to_string(window->dim()) + "d";
    }
    // A warmup call can carry more window steps than tokens; only the first
    // seqlen rows are touched.
    if (window->size(1) < seqlen || window->size(2) != dim ||
        window->size(3) != kConvWidth - 1) {
      return "window shape=" + std::to_string(window->size(0)) + "," +
             std::to_string(window->size(1)) + "," +
             std::to_string(window->size(2)) + "," +
             std::to_string(window->size(3));
    }
    if (!window_indices.has_value() ||
        window_indices->scalar_type() != at::kInt) {
      return "window needs int32 intermediate_state_indices";
    }
  }
  if (!nxt.has_value()) {
    if (sib.has_value() || parent.has_value()) {
      return "sibling/parent passed without retrieve_next_token";
    }
    return std::nullopt;
  }
  if (!sib.has_value() || !parent.has_value()) {
    return "tree arrays incomplete";
  }
  if (nxt->scalar_type() != at::kInt && nxt->scalar_type() != at::kLong) {
    return std::string("tree index dtype=") + c10::toString(nxt->scalar_type());
  }
  for (const auto& other : {&sib, &parent}) {
    if ((*other)->scalar_type() != nxt->scalar_type() ||
        (*other)->sizes() != nxt->sizes()) {
      return "tree arrays must share dtype and shape";
    }
  }
  if (nxt->size(1) < seqlen) {
    return "tree width=" + std::to_string(nxt->size(1)) +
           " < seqlen=" + std::to_string(seqlen);
  }
  return std::nullopt;
}

std::optional<std::string> gdn_conv1d_unsupported_reason(
    const at::Tensor& x, const at::Tensor& conv_state, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& cache_seqlens,
    const std::optional<at::Tensor>& conv_state_indices,
    const std::optional<at::Tensor>& num_accept_tokens,
    const std::optional<at::Tensor>& window,
    const std::optional<at::Tensor>& window_indices,
    const std::optional<at::Tensor>& nxt,
    const std::optional<at::Tensor>& sib,
    const std::optional<at::Tensor>& parent,
    const std::optional<int64_t>& pad_slot_id) {
  return conv_unsupported_reason_impl(x, conv_state, weight, bias,
                                      cache_seqlens, conv_state_indices,
                                      num_accept_tokens, window, window_indices,
                                      nxt, sib, parent, pad_slot_id);
}

std::optional<at::Tensor> gdn_conv1d_update(
    const at::Tensor& x, const at::Tensor& conv_state, const at::Tensor& weight,
    int64_t stream, const std::optional<at::Tensor>& bias, bool silu,
    const std::optional<at::Tensor>& cache_seqlens,
    const std::optional<at::Tensor>& conv_state_indices,
    const std::optional<at::Tensor>& num_accept_tokens,
    const std::optional<at::Tensor>& window,
    const std::optional<at::Tensor>& window_indices,
    const std::optional<at::Tensor>& nxt,
    const std::optional<at::Tensor>& sib,
    const std::optional<at::Tensor>& parent,
    const std::optional<int64_t>& pad_slot_id) {
  auto reason = conv_unsupported_reason_impl(
      x, conv_state, weight, bias, cache_seqlens, conv_state_indices,
      num_accept_tokens, window, window_indices, nxt, sib, parent, pad_slot_id);
  if (reason.has_value()) {
    return std::nullopt;
  }
  expect_same_device(x, conv_state, "conv_state");

  DeviceGuard guard(x.device().index());
  const bool unsqueeze = x.dim() == 2;
  at::Tensor x3 = unsqueeze ? x.unsqueeze(-1) : x;
  const int64_t batch = x3.size(0);
  const int64_t dim = x3.size(1);
  const int64_t seqlen = x3.size(2);
  at::Tensor out = at::empty_like(x3);
  const bool tree = nxt.has_value();
  const bool save_inter = window.has_value();
  const int64_t pad = pad_slot_id.value_or(0);

  auto stride2 = [](const std::optional<at::Tensor>& t) {
    return t.has_value() ? std::pair<int64_t, int64_t>{t->stride(0), t->stride(1)}
                         : std::pair<int64_t, int64_t>{0, 0};
  };
  const auto nxt_strides = stride2(nxt);
  const auto sib_strides = stride2(sib);
  const auto par_strides = stride2(parent);
  const int64_t idx64 =
      (!tree || nxt->scalar_type() == at::kLong) ? 1 : 0;
  const int64_t s_ii = window_indices.has_value() ? window_indices->stride(0) : 0;
  int64_t win_strides[4] = {0, 0, 0, 0};
  if (window.has_value()) {
    for (int i = 0; i < 4; ++i) {
      win_strides[i] = window->stride(i);
    }
  }

  gdn_causal_conv1d_update(
      x3.data_ptr(), weight.data_ptr(), data_or_null(bias),
      conv_state.data_ptr(), data_or_null(conv_state_indices),
      window.has_value() ? window->data_ptr() : nullptr,
      data_or_null(window_indices), data_or_null(nxt), data_or_null(sib),
      parent.has_value() ? parent->data_ptr() : nullptr, out.data_ptr(),
      static_cast<int>(batch), static_cast<int>(dim), static_cast<int>(seqlen),
      static_cast<int>(conv_state.size(2)),
      static_cast<int>(conv_state.size(0)), static_cast<int>(x3.stride(0)),
      static_cast<int>(x3.stride(1)), static_cast<int>(x3.stride(2)),
      static_cast<int>(weight.stride(0)), static_cast<int>(weight.stride(1)),
      static_cast<int>(conv_state.stride(0)),
      static_cast<int>(conv_state.stride(1)),
      static_cast<int>(conv_state.stride(2)),
      static_cast<int>(conv_state_indices.has_value()
                           ? conv_state_indices->stride(0)
                           : 0),
      static_cast<int>(win_strides[0]), static_cast<int>(win_strides[1]),
      static_cast<int>(win_strides[2]), static_cast<int>(win_strides[3]),
      static_cast<int>(s_ii), static_cast<int>(nxt_strides.first),
      static_cast<int>(nxt_strides.second), static_cast<int>(sib_strides.first),
      static_cast<int>(sib_strides.second), static_cast<int>(par_strides.first),
      static_cast<int>(par_strides.second), static_cast<int>(out.stride(0)),
      static_cast<int>(out.stride(1)), static_cast<int>(out.stride(2)),
      static_cast<int>(pad), bias.has_value() ? 1 : 0, silu ? 1 : 0,
      conv_state_indices.has_value() ? 1 : 0, save_inter ? 1 : 0,
      tree ? 1 : 0, pad_slot_id.has_value() ? 1 : 0, static_cast<int>(idx64),
      stream_handle(stream));
  check_launch();
  return unsqueeze ? out.squeeze(-1) : out;
}

}  // namespace

void register_gdn(pybind11::module& m) {
  m.def("gdn_norm_pass", &gdn_norm_pass, py::arg("q"), py::arg("k"),
        py::arg("a"), py::arg("b"), py::arg("A_log"), py::arg("dt_bias"),
        py::arg("qn"), py::arg("kn"), py::arg("eg"), py::arg("beta"),
        py::arg("stream"), py::arg("hv") = py::none(),
        "q/k l2norm + eg/beta gating; fills qn/kn/eg/beta in place");
  m.def("gdn_verify", &gdn_verify, py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("a"), py::arg("b"), py::arg("A_log"), py::arg("dt_bias"),
        py::arg("cu_seqlens"), py::arg("ssm_states"), py::arg("cache_indices"),
        py::arg("snap_flat"), py::arg("snap_idx"), py::arg("cache_steps"),
        py::arg("draft_token_num"), py::arg("o"), py::arg("stream"),
        py::arg("prev_step") = py::none(),
        py::arg("store") = py::none(), py::arg("store_row") = py::none(),
        py::arg("prenorm") = false, py::arg("prenorm_gate") = true,
        py::arg("qn") = py::none(), py::arg("kn") = py::none(),
        py::arg("eg") = py::none(), py::arg("beta") = py::none(),
        "B+ verify: writes the snapshot row selected by store/store_row");
  m.def("gdn_verify_full", &gdn_verify_full, py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("a"), py::arg("b"), py::arg("A_log"),
        py::arg("dt_bias"), py::arg("cu_seqlens"), py::arg("ssm_states"),
        py::arg("cache_indices"), py::arg("snap_flat"), py::arg("snap_idx"),
        py::arg("cache_steps"), py::arg("draft_token_num"), py::arg("o"),
        py::arg("stream"), "Full-snapshot verify (A/B reference)");
  m.def("gdn_compensate", &gdn_compensate, py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("a"), py::arg("b"), py::arg("A_log"),
        py::arg("dt_bias"), py::arg("ssm_states"), py::arg("h0_rows"),
        py::arg("snap_flat"), py::arg("snap_idx"), py::arg("cache_steps"),
        py::arg("store_step"), py::arg("n_seq"), py::arg("tok_per_seq"),
        py::arg("o"), py::arg("stream"),
        "Replay state only; slots with h0_rows < 0 exit");
  m.def("gdn_compensate_multi", &gdn_compensate_multi, py::arg("layer_ptrs"),
        py::arg("h0_indices"), py::arg("snap_idx"), py::arg("cache_steps"),
        py::arg("store_step"), py::arg("n_seq"), py::arg("tok_per_seq"),
        py::arg("hv"), py::arg("h"), py::arg("kdim"), py::arg("vdim"),
        py::arg("stream"),
        "One-launch state replay over an int64 [L, 15] pointer table");
  m.def("gdn_conv1d_unsupported_reason", &gdn_conv1d_unsupported_reason,
        py::arg("x"), py::arg("conv_state"), py::arg("weight"), py::arg("bias"),
        py::arg("cache_seqlens") = py::none(),
        py::arg("conv_state_indices") = py::none(),
        py::arg("num_accept_tokens") = py::none(),
        py::arg("intermediate_conv_window") = py::none(),
        py::arg("intermediate_state_indices") = py::none(),
        py::arg("retrieve_next_token") = py::none(),
        py::arg("retrieve_next_sibling") = py::none(),
        py::arg("retrieve_parent_token") = py::none(),
        py::arg("pad_slot_id") = -1);
  m.def("gdn_conv1d_update", &gdn_conv1d_update, py::arg("x"),
        py::arg("conv_state"), py::arg("weight"), py::arg("stream"),
        py::arg("bias") = py::none(),
        py::arg("silu") = false, py::arg("cache_seqlens") = py::none(),
        py::arg("conv_state_indices") = py::none(),
        py::arg("num_accept_tokens") = py::none(),
        py::arg("intermediate_conv_window") = py::none(),
        py::arg("intermediate_state_indices") = py::none(),
        py::arg("retrieve_next_token") = py::none(),
        py::arg("retrieve_next_sibling") = py::none(),
        py::arg("retrieve_parent_token") = py::none(),
        py::arg("pad_slot_id") = -1,
        "HIP causal_conv1d update; None when the caller must stay on Triton");
}

}  // namespace lightop::gdn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  lightop::gdn::register_gdn(m);
}
