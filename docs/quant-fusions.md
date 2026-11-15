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
