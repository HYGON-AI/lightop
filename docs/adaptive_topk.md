# Adaptive Top512 / Top1024 / Top2048

The implementation lives in `csrc/topk/fuse_topk_transform.cu` and
`csrc/topk/adaptive_topk.h`. Its GPU entry point is
`at::native::adaptive_topk_kernel<K, PhysicalPages>`.

Build LightOp normally. On wave64 HIP, `SGL_USE_LIGHTOP_TOPK_BACKAND=0`, an empty
value, or an unset variable selects the native adaptive kernel by default:

| Interface | K | Output transform |
| --- | --- | --- |
| `lightop.attention.fast_topk_transform_fused` | 2048 | page-size-1 table and cumulative query groups |
| `lightop.attention.fast_topk_transform_ragged_fused` | 2048 | logical ID plus per-row offset |
| `lightop.attention.topk_transform_512` | 512, or 1024 by output width | physical page ID and in-page offset, optional raw IDs |
| `lightop.sampling.topk_transform_1024` | 1024 | physical page ID and in-page offset, optional raw IDs |

No import hook, ctypes library, extra Python package, or SGLang source change is
required. Backend `1` retains the previous 8-bit radix implementation for all
three K values; backend `2` retains the previous VGPR implementation for fixed
Top512 only. Backend `2` still rejects Top1024 and paged/ragged Top2048.
The choice is cached per process, so set it before the first TopK call.
The experimental `LIGHTOP_ADAPTIVE_TOPK` variable is removed and has no effect.
CUDA builds retain their existing backend behavior; the new default is HIP-only.

For SGLang's ordinary LightOp paged-MQA/TopK path:

```bash
export SGL_USE_LIGHTOP_TOPK_BACKAND=0  # Or unset this variable.
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK=0
```

The caller must provide fully written FP32 scores over every valid row range.
Sparse-mask producer output with unwritten implicit zeros is not supported.
The separate sparse-mask and KPool APIs are unchanged. Existing launch scripts
that explicitly export backend `1` still select backend `1`; remove that export
or change it to `0` to use adaptive selection.
Fixed Top512/1024 preserve length clamping to score/page-table capacity, negative
physical-page sentinels, noncontiguous row strides, and optional raw-index output.

## Selection

The kernel uses one 1024-thread block per row on wave64 HIP (validated on gfx936).
Rows with at most K entries copy all valid indices and pad with -1. Top2048
rows use sampling from 8192 entries; Top512/Top1024 retain the 32768-entry
threshold. Eligible rows sample 2048 evenly spaced scores to propose a cutoff
that aims for 1.5*K candidates. A complete scan counts all scores above/equal to the
cutoff and caches at most 2*K strictly greater candidates.

| K | Candidate target | Cache capacity |
| --- | --- | --- |
| 512 | 768 | 1024 |
| 1024 | 1536 | 2048 |
| 2048 | 3072 | 4096 |

- If the cutoff is the exact boundary, retain all greater scores and the needed
  equal-score entries in logical-index order.
- If the complete candidate set fits and contains TopK, refine that set.
- Otherwise discard the proposal and select from the complete valid row.

Short rows and rejected proposals use a coarse monotone FP16 partition, followed
by full FP32-key refinement of the boundary bucket. Buckets exceeding capacity
fall back to full-row refinement; no candidate or bucket is silently truncated.
NaNs take the full ordered-FP32-key path. The FP32 bit ordering includes signed
zero and NaN payloads, matching the prototype's explicit total-order contract.

Selected logical IDs are sorted ascending before page-table mapping. Equal keys
prefer smaller logical IDs. This can select different tied tokens from the
legacy atomic implementation; exact operator selection is not a model accuracy
certificate. Sampling can cost extra when it frequently falls back, so speedup
is workload dependent. All row-dependent decisions execute on GPU at replay.

## Validation

```bash
env -u SGL_USE_LIGHTOP_TOPK_BACKAND python3 -m pytest -q test/test_adaptive_topk.py test/test_topk_backend_selection.py
```

Tests cover all three K values, CPU-reference selection, sparse/zero/negative/narrow distributions,
misleading samples, row starts, strided storage, query groups, padding, graph
replay with changing values, physical pages, ragged offsets, and streams/devices. Build and profiling artifacts
belong outside the source tree's tracked files.
