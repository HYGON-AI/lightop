# K3 EP SiTU scheduling

`fuse_situ_mul_quant_ep` keeps its public signature, SiTU activation, FP32
intermediates, INT8 rounding, scale calculation and output layout. Existing
framework calls need no changes. Both scheduling switches below default on.

## Dispatch

The optimized paths require ROCm, BF16 input, 28 experts and hidden width 3072
(input width 6144). `expect_m` is a scheduling hint, not a correctness bound.

| Hint / actual device counts | Schedule |
| --- | --- |
| Hint 1–4, fewer than 512 valid rows in total | Pack valid rows across 128 CTAs, 256 threads each |
| Hint 1–4, at least 512 valid rows in total | 256 CTAs, four waves per CTA, one wave per row |
| Hint 5–19 | Per-expert grid with tuned count loading and max reduction |
| Other dtype, shape, platform or hint | Original grid and kernel body |

The early path reads the 28 device counts and builds their prefix in registers.
Ballot and shuffle map packed row numbers to expert/token pairs. The wave path
removes block-wide barriers. The steady path avoids the shared count broadcast
and simplifies the max reduction without changing its result.

Every path processes the actual valid count, even when one expert receives
thousands of rows and the hint is one. Counts are read at execution/replay time;
there is no host readback, scratch allocation, atomic work queue or extra launch.
Padding rows are untouched. The Python wrapper initializes newly allocated
padding scales to one. Counts retain the existing contract:
`0 <= masked_m[e] <= tokens_per_expert`.

## Same-wheel comparison

- `LIGHTOP_SITU_EP_PACKED=0`: original scheduling.
- `LIGHTOP_SITU_EP_PACKED=1 LIGHTOP_SITU_EP_TUNED=0`: earlier packed path for hints
  1–4; original scheduling otherwise.
- Both unset or `=1`: new tuned paths above.

The switches are cached on the first call. Use a fresh process when changing
them. Optimized profiler names contain `situ_mul_quant_ep_kernel_tuned<true>`
for the early packed path or `<false>` for the tuned per-expert path.

## Correctness checks

```bash
LIGHTOP_SITU_EP_PACKED=0 LIGHTOP_SITU_EP_TUNED=0 python3 -m pytest -q test/test_situ_ep_packed_schedule.py
LIGHTOP_SITU_EP_PACKED=1 LIGHTOP_SITU_EP_TUNED=0 python3 -m pytest -q test/test_situ_ep_packed_schedule.py
LIGHTOP_SITU_EP_PACKED=1 LIGHTOP_SITU_EP_TUNED=1 python3 -m pytest -q test/test_situ_ep_packed_schedule.py
```

The standalone test checks an independent FP32 reference, untouched padding,
changed counts after graph capture, the 512-row transition in both directions,
and hint boundaries. The reference permits a one-code INT8 difference from fast
sigmoid approximation; this is separate from bitwise comparison to the previous
LightOp implementation.

The validated October 11 build additionally passed bitwise output/scale checks
on 261 cases per mode, 24 original-wheel goldens, 36 dynamic graph cases per
mode, non-default streams, dtype/shape fallbacks, and framework registration.
The same binary was smoke-tested on eight nodes. These are prior GPU results;
source migration checks do not constitute a new GPU or full-model accuracy run.

## Measured scope

On gfx938 (BW1300_LC), the same-wheel comparison used `PACKED=1`, switching
`TUNED=0/1`. FP32 activation and quantization math were unchanged.

| Workload | Earlier packed path | Tuned path |
| --- | ---: | ---: |
| BF16 E28/H3072, one expert with 2048 valid rows, hint 1 | 135.47 µs | 83.35 µs |
| Same, 4096 valid rows | 264.52 µs | 159.21 µs |
| Actual TP32/DP16/EP32 verify, local BS4, weighted SiTU kernel mean | 58.116 µs | 57.189 µs |

Synthetic timings include scale initialization and use warmed graph replay
with rotating modes. The all-empty synthetic case increased from approximately
4.71 to 5.20 µs. Two end-to-end runs per mode were effectively flat in overall
throughput; they do not establish a significant whole-model performance gain.
