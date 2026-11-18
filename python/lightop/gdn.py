# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""GDN recurrent HIP kernel.

Semantic wrappers around ``lightop._gdn``: callers pass tensors plus the few
scalars that are not derivable from their shapes. Strides, head ratios, the
softplus literals and the verify store rows are computed in the binding, and a
call never names a C entry point.

The recurrence is bit-exact with the aiter/Triton reference (``maxULP=0`` over
the recorded sweep); ``gdn_verify`` writes only the committed snapshot row
(B+), ``gdn_compensate`` replays the truncated slots, and
``gdn_compensate_multi`` does that for every layer in one launch.

``A_log`` and ``dt_bias`` are read as ``const float*`` by every entry: pass
bfloat16 tensors through :func:`f32` (or use the wrappers here, which cast for
you) or the kernel runs off the buffer and every head's decay is wrong.
"""

import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor

# Scratch for the prenorm verify path, one entry per size. Captures bake the
# buffer addresses, so the same size must return the same tensors.
_PRENORM_BUF = {}
# The binding is HIP-native (no c10/cuda host API), so the current torch stream
# is read here and passed as the opaque handle the C ABI takes.
_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _stream(device) -> int:
    index = device.index if isinstance(device, torch.device) else int(device)
    if _RAW_STREAM is not None:
        return int(_RAW_STREAM(index))
    return int(torch.cuda.current_stream(index).cuda_stream)


_EXT = None


def _gdn_arch():
    """The current device's architecture (``gfx938``), or None without a GPU."""
    try:
        name = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).gcnArchName
        return str(name).split(":")[0]
    except Exception:  # noqa: BLE001 - CPU hosts have no arch directory
        return None


def _candidates(lib_dir, arch):
    """Per-arch kernel first, then the flat layout older wheels shipped."""
    paths = []
    if arch:
        paths.extend(
            lib_dir / "gdn" / arch / ("_gdn" + suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        )
    paths.extend(
        lib_dir / ("_gdn" + suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )
    return paths


def _ext():
    """Load the prebuilt kernel module shipped under lib/ (see build_gdn.sh)."""
    global _EXT
    if _EXT is not None:
        return _EXT
    from .utils import resource_path

    lib_dir = Path(resource_path("lib"))
    for path in _candidates(lib_dir, _gdn_arch()):
        if path.is_file():
            spec = importlib.util.spec_from_file_location("lightop._gdn", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _EXT = module
            return _EXT
    raise RuntimeError(
        "lightop.gdn: no _gdn*.so under lib/gdn/<arch>/ or lib/; "
        "run scripts/build_gdn.sh"
    )


def f32(x: Tensor) -> Tensor:
    """Cached fp32 shadow of A_log/dt_bias for the kernel's float pointers."""
    if x.dtype == torch.float32:
        return x
    hit = getattr(x, "_lightop_gdn_f32", None)
    if hit is None or hit[1] != x._version:
        hit = (x.detach().float().contiguous(), x._version)
        try:
            x._lightop_gdn_f32 = hit
        except Exception:  # noqa: BLE001 - tensors that reject attributes
            pass
    return hit[0]


def prenorm_buffers(rows: int, h: int, kdim: int, hv: int,
                    device) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Persistent qn/kn/eg/beta scratch for the prenorm verify path."""
    key = (int(rows), int(h), int(kdim), int(hv), str(device))
    ent = _PRENORM_BUF.get(key)
    if ent is None:
        qn = torch.empty((rows, h, kdim), dtype=torch.float32, device=device)
        kn = torch.empty_like(qn)
        eg = torch.empty((rows, hv), dtype=torch.float32, device=device)
        beta = torch.empty_like(eg)
        ent = _PRENORM_BUF[key] = (qn, kn, eg, beta)
    return ent


def gdn_norm_pass(q: Tensor, k: Tensor, a: Tensor, b: Tensor, A_log: Tensor,
                  dt_bias: Tensor, qn: Tensor, kn: Tensor, eg: Tensor,
                  beta: Tensor, hv: Optional[int] = None):
    """q/k l2norm (dense fp32) plus eg/beta gating for each hv head."""
    return _ext().gdn_norm_pass(q, k, a, b, f32(A_log), f32(dt_bias), qn, kn,
                                eg, beta, hv=hv, stream=_stream(q.device))


def gdn_verify(q: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor,
               A_log: Tensor, dt_bias: Tensor, cu_seqlens: Optional[Tensor],
               ssm_states: Tensor, cache_indices: Tensor, snap_flat: Tensor,
               snap_idx: Tensor, cache_steps: int, draft_token_num: int,
               o: Tensor, prev_step: Optional[Tensor] = None,
               store: Optional[Tensor] = None,
               store_row: Optional[Tensor] = None, prenorm: bool = False,
               prenorm_gate: bool = True) -> Tensor:
    """Write the full-run snapshot to row (prev_step + 1) % cache_steps.

    ``store``/``store_row`` are per batch and default to the full-accept row
    (draft_token_num - 1). ``prenorm`` hoists the q/k l2norm and, with
    ``prenorm_gate``, the eg/beta gating into one launch of gdn_norm_pass.
    ``o`` is written in place and returned.
    """
    qn = kn = eg = beta = None
    if prenorm:
        hv = int(v.shape[-2])
        h = int(q.shape[-2])
        kdim = int(q.shape[-1])
        rows = max(int(snap_flat.shape[0]), int(q.shape[1]))
        qn, kn, eg, beta = prenorm_buffers(rows, h, kdim, hv, q.device)
    return _ext().gdn_verify(q, k, v, a, b, f32(A_log), f32(dt_bias), cu_seqlens,
                             ssm_states, cache_indices, snap_flat, snap_idx,
                             int(cache_steps), int(draft_token_num), o,
                             stream=_stream(q.device), prev_step=prev_step,
                             store=store, store_row=store_row,
                             prenorm=bool(prenorm),
                             prenorm_gate=bool(prenorm_gate), qn=qn, kn=kn,
                             eg=eg, beta=beta)


def gdn_verify_full(q: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor,
                    A_log: Tensor, dt_bias: Tensor, cu_seqlens: Optional[Tensor],
                    ssm_states: Tensor, cache_indices: Tensor, snap_flat: Tensor,
                    snap_idx: Tensor, cache_steps: int, draft_token_num: int,
                    o: Tensor) -> Tensor:
    """Full-snapshot entry: same kernel and inputs, every step written."""
    return _ext().gdn_verify_full(q, k, v, a, b, f32(A_log), f32(dt_bias),
                                  cu_seqlens, ssm_states, cache_indices,
                                  snap_flat, snap_idx, int(cache_steps),
                                  int(draft_token_num), o,
                                  stream=_stream(q.device))


def gdn_compensate(q: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor,
                   A_log: Tensor, dt_bias: Tensor, ssm_states: Tensor,
                   h0_rows: Tensor, snap_flat: Tensor, snap_idx: Tensor,
                   cache_steps: int, store_step: Tensor, n_seq: int,
                   tok_per_seq: int, o: Tensor) -> Tensor:
    """Replay state only (no o); slots with h0_rows < 0 exit immediately."""
    return _ext().gdn_compensate(q, k, v, a, b, f32(A_log), f32(dt_bias),
                                 ssm_states, h0_rows, snap_flat, snap_idx,
                                 int(cache_steps), store_step, int(n_seq),
                                 int(tok_per_seq), o,
                                 stream=_stream(q.device))


def gdn_compensate_multi(layer_ptrs: Tensor, h0_indices: Tensor,
                         snap_idx: Tensor, cache_steps: int,
                         store_step: Tensor, n_seq: int, tok_per_seq: int,
                         hv: int, h: int, kdim: int, vdim: int) -> Tensor:
    """One-launch state replay over an int64 [L, 15] layer pointer table.

    Columns, as the kernel reads them: q, k, v, a, b, A_log, dt_bias,
    h0_source, o, snap_buf, stride_q, stride_k, stride_v, stride_a, stride_b.
    """
    return _ext().gdn_compensate_multi(layer_ptrs, h0_indices, snap_idx,
                                       int(cache_steps), store_step, int(n_seq),
                                       int(tok_per_seq), int(hv), int(h),
                                       int(kdim), int(vdim),
                                       stream=_stream(layer_ptrs.device))


def _conv_silu(activation) -> bool:
    return activation is True or activation in ("silu", "swish")


def gdn_conv1d_unsupported_reason(
        x: Tensor, conv_state: Tensor, weight: Tensor,
        bias: Optional[Tensor] = None, cache_seqlens: Optional[Tensor] = None,
        conv_state_indices: Optional[Tensor] = None,
        num_accept_tokens: Optional[Tensor] = None,
        intermediate_conv_window: Optional[Tensor] = None,
        intermediate_state_indices: Optional[Tensor] = None,
        retrieve_next_token: Optional[Tensor] = None,
        retrieve_next_sibling: Optional[Tensor] = None,
        retrieve_parent_token: Optional[Tensor] = None,
        pad_slot_id: Optional[int] = -1) -> Optional[str]:
    """Short reason this call must stay on Triton; None when HIP can run it."""
    return _ext().gdn_conv1d_unsupported_reason(
        x, conv_state, weight, bias, cache_seqlens, conv_state_indices,
        num_accept_tokens, intermediate_conv_window,
        intermediate_state_indices, retrieve_next_token,
        retrieve_next_sibling, retrieve_parent_token, pad_slot_id)


def gdn_conv1d_update(
        x: Tensor, conv_state: Tensor, weight: Tensor,
        bias: Optional[Tensor] = None, activation=None,
        cache_seqlens: Optional[Tensor] = None,
        conv_state_indices: Optional[Tensor] = None,
        num_accept_tokens: Optional[Tensor] = None,
        intermediate_conv_window: Optional[Tensor] = None,
        intermediate_state_indices: Optional[Tensor] = None,
        retrieve_next_token: Optional[Tensor] = None,
        retrieve_next_sibling: Optional[Tensor] = None,
        retrieve_parent_token: Optional[Tensor] = None,
        pad_slot_id: Optional[int] = -1) -> Optional[Tensor]:
    """HIP causal_conv1d update; None when the caller must stay on Triton.

    ``x`` may be (batch, dim, seqlen) or (batch, dim) for a single token.
    When ``intermediate_conv_window`` is given the conv history is also stored
    there, one int32 row per batch from ``intermediate_state_indices``.
    """
    return _ext().gdn_conv1d_update(
        x, conv_state, weight, stream=_stream(x.device), bias=bias,
        silu=_conv_silu(activation), cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
        num_accept_tokens=num_accept_tokens,
        intermediate_conv_window=intermediate_conv_window,
        intermediate_state_indices=intermediate_state_indices,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_parent_token=retrieve_parent_token,
        pad_slot_id=pad_slot_id)
