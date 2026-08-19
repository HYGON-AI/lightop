# `gather_and_maybe_dequant_cache`

## Purpose

`gather_and_maybe_dequant_cache` gathers token rows from a paged KV cache into
a contiguous workspace.  In FP8 cache modes it also decodes each one-byte FP8
value, multiplies it by `scale`, and stores the result in the dtype selected by
`dst`.

The cache addressing and launch configuration were migrated from clean vLLM
v0.21.0 commit `8c979d2e4bad77867ea1d5c22af5e6e34314aafe`.  LightOp adds E5M2 to the
same output dispatch used by E4M3.

## Public interfaces

The primary public Python interface follows LightOp's traditional top-level
wrapper layout:

```python
from lightop import gather_and_maybe_dequant_cache

gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None
```

Its argument order, default value, and in-place `dst` mutation match vLLM's
Python wrapper. The exact same wrapper is also re-exported from
`lightop.kvcache` for categorized imports. The low-level PyBind entry remains
available as `lightop.op.gather_and_maybe_dequant_cache` for native API
diagnostics.

## Contract

| Argument | Contract |
| --- | --- |
| `src_cache` | GPU tensor shaped `[num_blocks, block_size, entries...]`; entry dimensions must be contiguous |
| `dst` | GPU FP32, FP16, or BF16 tensor shaped `[num_tokens, head_dim]` |
| `block_table` | GPU int32 tensor shaped `[batch, block_indices]` with contiguous rows |
| `cu_seq_lens` | Contiguous GPU int32 tensor shaped `[batch + 1]` |
| `token_to_seq` | Contiguous GPU int32 tensor with at least `num_tokens` entries |
| `num_tokens` | Non-negative value within int32 range |
| `kv_cache_dtype` | See the table below |
| `scale` | Contiguous, single-element GPU FP32 tensor |
| `seq_starts` | Optional contiguous GPU int32 tensor with one entry per sequence |

The optimized kernel currently supports head dimensions 320 and 576.  It
writes to `dst` in place and returns `None`.

| `kv_cache_dtype` | Behavior |
| --- | --- |
| `auto`, `float16`, `bfloat16` | Gather without conversion; `src_cache` and `dst` must have the same dtype |
| `fp8`, `fp8_e4m3`, `fp8_ds_mla` | Decode E4M3 one-byte storage, multiply by `scale`, store to `dst` |
| `fp8_e5m2` | Decode E5M2 one-byte storage, multiply by `scale`, store to `dst` |

E4M3 and E5M2 use the same LightOp/DTK format selection.  When
`HIP_FP8_TYPE_OCP` is enabled, the HIP OCP types are used; otherwise the HIP
FNUZ types are used.  This matches the existing E4M3/E5M2 implementation in
`csrc/fuse_gather_cache_fp8_convert.cu`.

## Validation

Build and install a new wheel before running tests because the implementation
contains native HIP code.

Functional and interface validation:

```bash
pytest -sv test/test_gather_and_maybe_dequant_cache.py \
  test/test_kvcache_existing_api.py
```

The functional matrix covers:

- `auto`, E4M3, and E5M2 cache modes;
- FP32, FP16, and BF16 output;
- head dimensions 320 and 576;
- scales 1.0 and 1.5;
- absent and non-zero `seq_starts`;
- the current default GPU, following the LightOp test convention;
- unsafe input contracts and dtype aliases;
- the original large vLLM workload.

E5M2 expected values are produced by an independent raw-byte decoder, covering
all 256 byte encodings rather than reusing the native conversion routine.

Combined correctness and performance validation, following LightOp's common
single-file operator-test layout:

```bash
mkdir -p test-results
python test/test_gather_and_maybe_dequant_cache.py 2>&1 \
  | tee test-results/gather_cache_e5m2.log
```

The standalone script first executes the complete functional matrix, then
prints environment and device details and reports a Markdown performance
table. The timing section covers six cases: `auto`, E4M3, and E5M2 for head
dimensions 320 and 576. Each result is the median of five rounds, with ten
warmups and fifty kernel launches per round. The same file remains directly
collectable by pytest when only functional regression coverage is wanted.

## Compile integration

The Python wrapper lives at
`python/lightop/gather_and_maybe_dequant_cache.py`, matching the most common
LightOp layout in which a top-level module calls `lightop.op`. The
`lightop.kvcache` package only re-exports this same object and does not keep a
second implementation.

`setup.py` automatically includes repository-level `csrc/*.cu`, so
`csrc/gather_and_maybe_dequant_cache.cu` is compiled into `lightop.op`.  The
native symbol is included in `compat/lightop_op_symbols.txt`, and the KV-cache
category runner includes the operator's combined correctness and performance
test module.

PyTorch Dispatcher/FakeTensor registration is intentionally not added here.
The shared `lightop.ops.registration` and `lightop.ops.fake_impl` modules are
still migration placeholders.  The current interface supports normal eager
execution; Dispatcher registration should be added as a separate repository-
wide integration step if a caller requires `torch.compile` or FakeTensor.
