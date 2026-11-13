# MiniMax prefill operator validation

Correctness tests run by the existing CI workflow live in `test/ci_tests/`:

```bash
cd test/ci_tests
python3 -m pytest -v test_minimax_cp_packet.py test_minimax_qk_dispatch.py \
  test_minimax_qk_norm_rope.py test_minimax_public_api.py \
  test_swiglu_oai_fp8_quant_contiguous.py
```

Tests import the installed LightOp package, including the CPU dispatch test.
They do not require the `python/` source directory after CI sparse checkout.
GPU tests cover independent numerical oracles, changed-input graph replay,
layout/empty/tail cases, and invalid K/V scales (including a finite double
that overflows float). CPU API tests check classification aliases and missing
bindings. Normal full builds expose all four operators in `lightop.op`.
Private incremental-validation modules are not part of the public fallback
contract; callers with older wheels receive an explicit missing-binding error.

## Opt-in performance measurement

Performance tools are separate from correctness CI because useful latency
comparisons require an idle, identified GPU and stable clocks. There is no
hardware-dependent speedup assertion in default CI. Install a wheel built
from the revision under test, then run from the repository root:

```bash
HIP_VISIBLE_DEVICES=0 python3 test/benchmarks/benchmark_minimax_prefill.py \
  --rows 1 16 128 2048 8192 --mode graph --warmup 20 \
  --iterations 100 --repeats 5 --output /tmp/minimax-prefill-graph.json

HIP_VISIBLE_DEVICES=0 python3 test/benchmarks/benchmark_minimax_prefill.py \
  --rows 1 16 128 2048 8192 --mode eager --warmup 20 \
  --iterations 100 --repeats 5 --output /tmp/minimax-prefill-eager.json
```

`--rows` is the **local operator token-row count**, not request concurrency.
Use actual per-rank profile shapes for additional runs. Output paths must be
new; existing measurements are never overwritten. Results include all timing
samples, medians, shape/scale, framework/native module paths, device and HIP
metadata. Record the built commit and wheel hash alongside the JSON.

| Public API | Classified import | Unfused timing baseline |
| --- | --- | --- |
| `minimax_qk_norm_rope` | `lightop.attention` | Four FP32 Gemma RMSNorm chains, BF16 intermediate rounding, rotary64 and concatenation |
| `minimax_cp_fp8_pack` | `lightop.quant` | BF16 K/V scaling, clamp and E4M3FN casts, concatenate with exact BF16 index bytes |
| `minimax_cp_cache_scatter` | `lightop.kvcache` | Rank-major packet reorder followed by three indexed cache writes |
| `swiglu_oai_fp8_quant_contiguous` | `lightop.activation` | OAI activation, BF16 rounding, row amax/scale and E4M3FN quantization |

Top-level `lightop.*` aliases remain supported. QK uses packed width9856,
head_dim128 and rotary64; pack uses K512/V512/index128; OAI uses split
`[gate | up]` width6144, alpha1.702, beta1, limit7. Scatter timing uses unique
positive slots; the correctness suite separately checks slot-zero behavior.

Each case checks its outputs before timing. Graph mode also checks captured
outputs after replay. QK uses the same bounded BF16 tolerances as the oracle
tests; OAI compares scale and dequantized values, allowing FP32 reciprocal
rounding differences in the unfused chain. Packet and scatter checks are exact
bytes. The stronger independent FP64 OAI oracle is in the correctness suite,
not the performance baseline.

Timing uses synchronized GPU events after warmup; alternating baseline/fused
order reduces order bias. Eager GPU-event intervals can include host launch
gaps, and graph mode includes graph replay costs. Both paths include their
output/intermediate allocation behavior (graph uses its captured memory pool).
These are comparisons with unfused PyTorch operations, **not** proof of a gain
over another optimized kernel or of model TTFT/TPOT improvement. This tool does
not measure CP communication. No latency or end-to-end gain is claimed without
hardware measurements for the stated setup.
