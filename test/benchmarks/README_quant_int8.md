# INT8 quantization benchmark

This opt-in benchmark covers per-token quantization and all three OAI INT8 public APIs. It requires a HIP LightOp installation, PyTorch and Triton; it does not import SGLang or vLLM.

## Reproduce

Run from the repository root after building/installing LightOp:

```bash
python3 test/benchmarks/benchmark_quant_int8.py --output quant_int8.json
python3 -m pytest -q test/ci_tests/test_swiglu_oai_int8_import.py test/ci_tests/test_swiglu_oai_int8_quant.py test/ci_tests/test_per_token_quant_int8_wide.py
```

Defaults: seed 42, 20 warmups, 50 calls per captured graph, 7 alternating-order repeats. GPU events time one graph replay and divide by 50. Compiling, table construction, reference calculations and host allocation are outside timing. Buffers stay warm; this is an amortized API GPU measurement, not an individual end-to-end kernel duration. Raw samples and artifact/source hashes are in [the recorded results](results/quant_int8_gfx938_20261106.json).

## Numerical contracts

- Per-token results and scales match the preserved original Triton implementation exactly for all measured inputs. The CI tests also cover all finite BF16 encodings, tie cases, tiny/zero rows, unsupported layouts and changed-input graph replay.
- OAI uses split gate/up, FP32 activation with alpha 1.702 / beta 1 / clamp 7, BF16 rounding, then symmetric INT8. An independent PyTorch oracle checks scale rtol 1e-6 / atol 1e-12 and at most one INT8 level at <=1e-4 of live elements (FP32 sigmoid/BF16 boundary rounding).
- The regular masked entry and Triton reference write q=0/scale=1 to padding. masked_flat leaves padding unspecified and requires a consumer that masks it. The benchmark verifies sentinel padding remains untouched; timing comparisons include this deliberate work difference.
- decode_sparse uses 16 device counts [0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,64],clamped to capacity. Full and skewed occupancy are also tested.

## Measured environment and scope

2026-11-06; BW1100, gfx938, 64 CU; DTK-26.04-blas-nightly-1019; PyTorch 2.11.0; HIP 6.3.26113. Base revision 19329dc484ecde51942712a18edbf59a2d1cdad9, implementation parent 4834c58d25fd7269ec1458455c58cee1825c5448 plus this review fix. The original per-token Triton implementation remains the base comparator; OAI has no base LightOp INT8 API, so a self-contained equivalent of the prior scalar Triton formula is supplied.

Validation used the existing native runtime and the separately built OAI/per-token extensions with the edited Python wrappers, not a new monolithic wheel. Native artifact hashes are recorded. The 78 targeted tests passed; 18 benchmark cases passed their numerical checks. DTK 26.10 and a fresh complete wheel build still require CI.

## Per-token timings

Median microseconds; native falls back to original Triton for small rows and width 12288. The benchmark records eligibility to avoid presenting fallback as a native result.

| Shape | Original Triton | Public segments | Public native |
| --- | ---: | ---: | ---: |
| [64, 3072] | 2.848 | 2.736 | 2.851 |
| [64, 6144] | 3.466 | 3.050 | 3.453 |
| [64, 12288] | 4.384 | 3.507 | 4.384 |
| [2048, 3072] | 24.730 | 14.781 | 17.974 |
| [2048, 6144] | 34.013 | 24.912 | 30.179 |
| [2048, 12288] | 51.718 | 46.413 | 51.738 |
| [6144, 3072] | 68.438 | 35.450 | 40.643 |
| [6144, 6144] | 94.544 | 66.042 | 77.910 |
| [6144, 12288] | 143.427 | 128.525 | 143.389 |

## OAI timings

| Shape / occupancy | Original Triton | Contiguous or masked | Masked flat |
| --- | ---: | ---: | ---: |
| [2048, 6144] / full | 84.662 | 54.746 | n/a |
| [8192, 6144] / full | 328.422 | 196.486 | n/a |
| [16384, 6144] / full | 653.334 | 383.136 | n/a |
| [16, 128, 6144] / full | 85.219 | 67.616 | 78.477 |
| [16, 128, 6144] / skewed | 84.086 | 67.062 | 42.605 |
| [16, 128, 6144] / decode_sparse | 84.090 | 66.582 | 13.642 |
| [16, 1024, 6144] / full | 654.566 | 500.624 | 578.112 |
| [16, 1024, 6144] / skewed | 653.900 | 278.240 | 293.696 |
| [16, 1024, 6144] / decode_sparse | 653.353 | 57.885 | 14.822 |

Masked flat is best for the sparse decode distribution, but is slower than the regular masked implementation at full occupancy. There is no claim of a universal winner. Similarly, isolated segments/native ranking is not a substitute for model profiling: the validated application selects native for prefill and original Triton for decode based on its end-to-end profiles.

In a separate same-runtime model comparison (MiniMax M3.1 W4A8, CP8/EP8, BS4, 16K new input + 128K cached prefix, output 1), per-token cumulative GPU time averaged across 8 ranks decreased 114.841 -> 62.344 ms. That is an operator cumulative-time result, not a claimed model speedup; communication/shared-expert overlap affects the complete profile.
