# Masked OAI SwiGLU FP8 microbenchmark

Run from a checkout with this PR's LightOP extension installed, on an idle HCU:

```bash
python test/benchmarks/benchmark_swiglu_oai_fp8_quant_masked.py \
  --mode graph --output masked_graph.json
python test/benchmarks/benchmark_swiglu_oai_fp8_quant_masked.py \
  --mode eager --output masked_eager.json
# Optional bounded launch-configuration comparison:
python test/benchmarks/benchmark_swiglu_oai_fp8_quant_masked.py \
  --mode graph --sweep --output masked_launch_sweep.json
```

The default cases use 16 **local experts**, capacities 64/128/256, width
3072, contiguous BF16 input `[16, capacity, 6144]`, E4M3FN output and one FP32
scale per row. Capacity is padded storage per expert, **not global batch size**.
Balanced counts are `capacity // 4` per expert; skewed counts concentrate the
same total work on four experts. JSON includes the exact counts and live rows.
Use `--width` and `--capacities` for other workloads.

Each variant is checked against an independent FP32 OAI oracle before timing:
gate upper clamp 7, up symmetric clamp 7, alpha 1.702, up offset 1, row scale
`max(absmax, 1e-10) / 448`. The check allows one E4M3 rounding step, verifies
signs and preserves sentinel-filled padding. Graph output is checked too.
Ordinary SiLU or an implementation rounding the activation to BF16 before
quantization is not an equivalent performance reference.

The comparison includes the public API and both native schedules on identical
inputs: general `(row_blocks=32, threads=256)` and E=16 flattened
`(row_blocks=256, threads=256)`. `--sweep` compares general row blocks
16/32/64 and flat row blocks 128/256/512, each with 64/128/256 threads. These
launch values describe different grid geometries; they are not equal block
counts. The script does not change dispatch policy or claim that the current
defaults are optimal for every shape or device.

Timing uses GPU events, 20 warmup calls, 100 operations per sample and seven
samples by default. Results include every sample, medians, and ratios to the
general 32/256 schedule. Graph mode captures 100 calls and divides replay time
by 100; eager mode submits 100 ordinary calls and may include GPU idle time due
to host submission. Variants rotate measurement order. Allocations, input/count
copies, correctness checks and graph capture are outside timing; buffers are
persistent and counts are read on device by the kernels. This is repeated-data
microbenchmark timing, not an end-to-end model or cold-cache measurement.

The JSON records arguments, seed, Torch/HIP/device information and the installed
extension path/hash. Compare results only with matching hardware, clocks,
inputs and execution mode; retain raw samples rather than reporting only the
best run. Run without another GPU workload. No performance threshold is imposed
on shared CI runners. This opt-in script requires only Torch/LightOP and the
Python standard library. The mandatory correctness/graph/storage tests remain
in `test/ci_tests/test_swiglu_oai_fp8_quant_masked.py`.

## Validation scope (2026-10-30)

Both native sources from PR head `867cf255` were compiled on gfx938 with
`-O3 --offload-arch=gfx938`, Torch 2.11.0, in a focused extension with the PR's
Python API. An existing full extension supplied unrelated package imports.
This validates the two compiled kernels, not a new full wheel; full-wheel
validation remains the DTK CI matrix. All **201 existing CI correctness
checks** passed. All 19 variants in each of six workloads passed the benchmark
checks in both eager and graph modes.

A concurrent model-loading process was discovered in the resource audit, so
latency samples from that run are not published as performance evidence. Run
this script on an uncontended GPU before making a speedup or no-regression
claim. The new benchmark leaves the API, kernels and production dispatch
unchanged. Its purpose is to make the schedule tradeoffs and launch choices
measurable rather than assume a universally optimal default. Reading live
counts on the CPU to choose a variant would violate the capture/replay
contract and is not proposed here.
