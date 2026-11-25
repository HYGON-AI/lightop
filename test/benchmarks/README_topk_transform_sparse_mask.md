# Sparse-mask TopK transform (DSA decode)

The public entries are `lightop.attention.fast_topk_transform_sparse_mask_fused`
(HIP kernel `topk_transform_decode_kernel_sparse_mask`) and
`lightop.attention.fast_topk_transform_sparse_mask_fused_v2` (Triton).

Both consume the paged-MQA indexer logits of a DSA decode step and return the
`(rows, 2048)` page-id tensor used by the sparse attention kernel:
masked-out positions behave like a `+0.0` score, and the selection follows the
two paths documented by the HIP kernel (fast path with implicit-zero padding,
generic path ranking every valid position).

After building and installing LightOp, run on an idle device:

    HIP_VISIBLE_DEVICES=0 python3 -m pytest -q test/ci_tests/test_topk_transform_sparse_mask_triton.py
    HIP_VISIBLE_DEVICES=0 python3 test/benchmarks/benchmark_topk_transform_sparse_mask.py
    HIP_VISIBLE_DEVICES=0 python3 test/benchmarks/benchmark_topk_transform_sparse_mask.py --density 0.25

`--density` sets the fraction of positions whose mask bit is set. Production
MQA logits are overwhelmingly `+0`, which keeps the HIP kernel on its two-pass
fast path; a density of 1.0 with random logits forces the seven-pass generic
path instead. Report both when comparing to the reference, because the two
paths have very different baselines.

The script prints a markdown table ready for a PR description. Timing uses
`torch.cuda.Event` with a warmup and a fixed repeat count; run it on an idle
device only, since a co-running service inflates the baseline by 20-30 %.

`LIGHTOP_SPARSE_MASK_TOPK_TRITON=1` makes integrations call the Triton entry
by default; unset it to fall back to the HIP kernel.
