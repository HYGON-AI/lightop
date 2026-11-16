# BF16 MiniMax router projection

The public entry is lightop.moe.bf16_router_gemm(x, weight).
It computes logits x @ weight.T for contiguous BF16 inputs with shapes
[64, 6144] and [128, 6144] on gfx938. Accumulation and output are FP32.
The caller handles scoring and top-k expert selection.

After building and installing LightOp, run on an idle gfx938 device:

    HIP_VISIBLE_DEVICES=0 python3 -m pytest -q test/ci_tests/test_minimax_router_gemm.py
    HIP_VISIBLE_DEVICES=0 python3 test/benchmarks/benchmark_minimax_router_gemm.py \
        --warmup 100 --iterations 200 --repeats 7 > router_benchmark.json

This benchmark is opt-in and runs separately from the default CI tests.
It validates all three implementations against a double-precision reference
before timing. The router uses the same tolerance as its CI numerical test.
The BF16-output baseline allows BF16 rounding error.

Timing covers the complete public call, including both partial GEMM and
reduction kernels. HIP events on the current stream measure batches of calls
after warmup. Eager mode includes Python dispatch effects on device scheduling.
Graph mode captures a batch of complete calls and times one replay per sample,\namortizing host launch overhead. Both modes
report median, minimum, maximum, standard deviation and all samples in
microseconds per call. The graph output remains alive throughout replay.

The baselines have explicit output types:

- torch_bf16_in_bf16_out computes BF16 matmul and returns BF16.
- torch_fp32_in_fp32_out_including_casts converts both operands to FP32
  inside the timed call and returns FP32.

These baselines expose the dtype and conversion costs. Microbenchmark speedups
do not establish end-to-end model gains or identical top-k routing decisions.
The JSON also records device, Torch/HIP versions, shapes, seed and errors.

## Validation sample

Measured on gfx938 / BW1101 with Torch 2.11.0 and HIP 6.3.26113.
The router extension was rebuilt from this source and tested through the public
Python API in a task-local runtime. The existing package extensions were reused.
The two router CI tests passed; a full wheel rebuild is covered separately by CI.
The seed is 42, with 100 warmup calls, 200 calls per sample and 7 samples.

| Operation | Eager median (us) | Graph median (us) | Graph stdev (us) | Max abs error |
| --- | ---: | ---: | ---: | ---: |
| router_bf16_in_fp32_out | 16.140 | 14.746 | 0.007 | 6.10352e-05 |
| torch_bf16_in_bf16_out | 10.210 | 9.150 | 0.009 | 0.96933 |
| torch_fp32_in_fp32_out_including_casts | 31.274 | 27.419 | 0.012 | 9.91821e-05 |

In this run the router is faster than the FP32 baseline including conversions,
but slower than the BF16-output matmul. The reported error is against the same
FP64 oracle. These measurements do not claim a speedup over BF16 matmul or an
end-to-end model improvement.
