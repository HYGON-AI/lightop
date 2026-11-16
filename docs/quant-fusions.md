# Transform and quantization fusions

These operators run in the HCU/ROCm LightOp extension. They combine memory-heavy
preparation steps for an existing matrix multiplication consumer; they do not
perform GEMM. All allocating wrappers preserve input storage.

## Build and install

Use the normal repository build described in the root README:

```bash
source /opt/dtk/env.sh
python3 setup.py install
```

The normal build includes the operators in `lightop.op`. For a manylinux wheel,
use `./scripts/build_hcu_wheel.sh`. Use the matching DTK, ROCm PyTorch and Python
ABI for the target environment.

For development against an older installation that lacks these symbols, the
focused builder compiles the same sources and bindings:

```bash
python3 scripts/build_quant_fusions.py --build-dir /tmp/lightop-transform-build --op transform-int8 --install
python3 scripts/build_quant_fusions.py --build-dir /tmp/lightop-strided-rms-build --op strided-rms --install
```

The `transform-int8` target contains both INT8 and FP8 transforms. Focused
extensions are used only for symbols missing from `lightop.op`; rebuild the full
package when updating operators already present in `lightop.op`. Source Python
wrappers must also be installed from this revision. Installation replaces the
focused extension file atomically; restart workers to load a changed extension.
The focused builder preserves explicit FP32 multiply/FMA boundaries and does
not compile during normal serving imports.

## APIs

| API | Input | Independent outputs |
| --- | --- | --- |
| `lightop.quant.rope_per_token_quant_int8` | BF16/FP16 `[T,H,D]`, complex64 frequencies, INT32/INT64 positions | INT8 `[T,G,H/G*D]`, FP32 `[T,G,1]`; optional row pitch |
| `lightop.quant.rope_per_token_quant_fp8` | Same head layout | E4M3fn `[T,G,H/G*D]` or `[G,T,H/G*D]`, FP32 scales in the same row layout |
| `lightop.quant.scale_per_token_quant_int8` | INT32 `[T,G,R]` or `[G,T,R]`, FP32 `[T,G,1]` input scales, G*R channel scales | Floating `[T,G,R]`, INT8 `[T,G*R]`, FP32 `[T,1]`; floating output is `None` with `materialize_output=False` |
| `lightop.quant.per_token_quant_fp8_strided` | BF16/FP16 logical `[T,G,R]`, token-major or group-major storage | E4M3fn `[T,G*R]`, FP32 `[T,1]` |
| `lightop.norm.rms_norm_strided_per_token_quant` | Aligned padded/sliced BF16/FP16 matrix, matching norm weight | Normalized source-dtype matrix, INT8 matrix, FP32 `[T,1]` |
| `lightop.norm.rms_norm_strided_per_token_fp8_quant` | Same matrix layout | Normalized source-dtype matrix, E4M3fn matrix, FP32 `[T,1]` |

RoPE supports positive non-overlapping head/token strides, unit inner stride,
even head/rotary dimensions and complete group widths up to 16384. Positions
must be valid frequency-table indices. INT32 scaling supports complete token
widths up to 16384. Strided RMSNorm requires aligned unit column stride and
column counts divisible by 16: at most 8192 for INT8 or 16384 for FP8. Zero rows
return correctly shaped empty outputs without a kernel launch.

Native `_out` bindings validate shapes, dtypes, devices, strides, alignment and
output overlap. Transform outputs must have storage independent of every input
and other output, including otherwise disjoint slices of one storage. All
kernels use the input device's current stream and need no host read of tensor
values or synchronization.

## Arithmetic and reuse

RoPE-to-INT8 rounds the rotary tail to the activation dtype before absmax and
uses half-away-from-zero INT8 rounding. INT32 scale-to-INT8 explicitly rounds
through `output_dtype` before full-token absmax and round-to-nearest-even INT8.
INT8 absmax has a `1e-10` floor. Omitting the floating output removes its buffer
and store, while retaining that dtype's rounding in registers.

FP8 transforms reuse the existing E4M3 conversion helpers, now shared in
`csrc/quant_fp8.cuh`. Their scales are `absmax / 448`; zero rows have zero scales.
Strided RMSNorm reuses the existing INT8/FP8 RMSNorm kernels, retaining reduction,
activation rounding and the FP8 RMSNorm scale floor. The source Q/KV allocation
is never modified by these wrappers.

The default RoPE thread policy depends on architecture, dtype, output layout and
T. Explicit `threads` overrides it. Uncalibrated combinations keep 256 threads;
the generic implementation supports shapes beyond calibrated T40/T48/T128.
`splits` in INT32 scale quantization optionally repeats read-only full-row
reductions across CTAs that write disjoint slices; it needs no shared scratch
or cross-CTA synchronization.

The included regression files cover numerical parity, padding preservation,
empty inputs, layouts, thread choices and changing-input graph replay. Their
presence does not imply that tests or a full wheel were built during the final
code review; see the paired PR descriptions for executed checks and evidence.


## Reproduce operator performance comparisons

The opt-in benchmark covers all six public APIs above. Install this revision's
Python package and native extension, plus Triton, on the target HCU first. It
has no SGLang, vLLM or pytest dependency and is not collected by CI pytest.

```bash
python3 test/benchmarks/benchmark_quant_fusions.py --tokens 40 48 --output /tmp/quant-fusions.json
python3 test/benchmarks/benchmark_quant_fusions.py --suite rope-int8 --tokens 40 48 64 96 128 --output /tmp/rope-int8.json
python3 test/benchmarks/benchmark_quant_fusions.py --suite rope-fp8 --tokens 40 48 64 96 128 --output /tmp/rope-fp8.json
python3 test/benchmarks/benchmark_quant_fusions.py --suite rms-fp8 --rms-widths 512 1536 3072 8192 16384 --output /tmp/rms-fp8.json
```

All cases use BF16 activations. The default RoPE input is `[T,64,512]`, with
8 groups and 64 rotary channels. The INT8 output pitch includes 128 padding
bytes per group by default; `--int8-pitch-padding 0` selects compact output.
FP8 RoPE covers token-major and group-major outputs, including the split
baseline's quantized data/scale layout copies. INT32 scaling uses `[T,8,1024]`
with both accumulator layouts and both floating-output materialization modes.
Strided FP8 quantization reads logical `[T,8,1024]` from padded token-major and
group-major storage. RMS cases use padded `[T,width+512]` rows sliced to width;
default widths are 1280, 1536 and 3072. In the `all` suite, widths above 8192
are benchmarked only for FP8 RMSNorm, matching the INT8 API's size limit.

The split RoPE stage is a standalone Triton inverse rotation into a persistent
BF16 buffer. Its immutable nonrotary prefix is initialized outside timing; each
call rewrites the tail from the original input rather than rotating an already
rotated tail. INT8 RoPE uses a separate half-away quantizer with matching output
pitch. It intentionally does not use generic `per_token_quant_int8`, whose RNE
rounding is different. INT32 scaling has a standalone multiply/multiply/BF16
store stage followed by the existing LightOp INT8 quantizer. FP8 split paths
reuse `per_token_quant_fp8`. RMS comparisons use a contiguous copy plus the
existing native fused RMSNorm/quant, so they measure removal of the input copy
while retaining the same normalization arithmetic and scale floor.

Before timing, and after graph replay, each pair must match bit for bit in
floating outputs, quantized outputs and scales. A mismatch aborts the run;
source and padding preservation are also checked. By default each variant gets
20 warmup calls, then a graph of 50 calls; GPU events measure 7 repeats with
alternating variant order. `--warmup`, `--iterations` and `--repeats` override
these counts. Compilation, initialization and correctness checks are outside
timing. Public allocating APIs are captured; replay has no Python allocation
work. These are repeated warm-buffer operator comparisons, not individual
production trace durations or an end-to-end service A/B.

The script prints and saves the LightOp package path/version, source revision,
PyTorch/Triton/HIP versions, device properties, native binding module paths and
SHA256 hashes. JSON includes every timing sample, median microseconds per call,
speedup and latency reduction. It preserves completed cases and records a
failed status if a later case fails. No pre-recorded performance numbers are
substituted for measurements from the installed build.
