# DSV4.1 indexed query preparation

`lightop.dsv41_indexer_prep.dsv41_indexer_rope_fake_quant(x, freqs_cis, positions)`
fuses frequency-table lookup with tail RoPE and per-32 UE8M0/E2M1 fake quantization
for the prefill indexer. It returns BF16 FP4-grid values, not packed FP4 bytes.

- `x`: contiguous GPU BF16 `[T, H, 128]`, with `H > 0`.
- `freqs_cis`: contiguous GPU complex64 `[P, 32]`.
- `positions`: contiguous GPU int32/int64 `[T]`; every value must be in `[0, P)`.
  Position values are a caller contract, without a host synchronization check.
- Inputs must share a device and are not modified. The output is newly allocated.
- The last 64 channels receive RoPE. The intermediate BF16 rounding and native
  FP4 ties-to-even/signed-zero behavior are retained.
- The implementation uses the current device/stream and supports CUDA/HIP graph
  capture and replay with updated inputs. It is an inference operator.

The extension is `lightop._dsv41_prep`, built by the regular LightOp `setup.py`.
It uses `-ffp-contract=off` to preserve the native real multiply/subtract, and an
explicit `fmaf` for the imaginary expression. Do not replace these with a blanket
fast-math transformation without repeating bitwise validation.

The kernel processes eight heads per CTA with four adjacent BF16 values per lane.
A width-8 reduction covers each 32-value quantization group. An unaligned,
contiguous BF16 storage offset uses scalar input loads.

## Validation and benchmark

The reference tests require the deployed SGLang `sgl_kernel.rope_fp4_fake_quant`
on a ROCm/HCU GPU:

```bash
python3 -m pytest -q test/ci_tests/test_dsv41_indexer_prep.py
python3 test/benchmarks/benchmark_dsv41_indexer_prep.py --tokens 512 1024 4096 8192 32768
```

The benchmark includes frequency gather in the native baseline, checks output
bits before timing, and reports CUDA event medians after warmup. It measures an
operator, not model TTFT. Prefill CP8 with a global 32768-token chunk uses 4096
local tokens per rank. Framework integration should remain opt-in and preserve
the original path for unsupported layouts.
