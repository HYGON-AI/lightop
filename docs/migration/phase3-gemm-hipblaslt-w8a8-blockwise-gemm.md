# GEMM migration: `hipblaslt_w8a8_blockwise_gemm`

The original LMSlim `quantize/quant_ops.py` is retained under
`lightop._lmslim_native`, together with its registration and AWQ config
dependencies. Project-local imports are redirected to private relative paths;
the function and native hipBLASLt call flow are unchanged.

The public API is a direct binding at
`lightop.gemm_ops.hipblaslt_w8a8_blockwise_gemm`. The focused test compares a
256-by-128-by-512 FP8 GEMM with an independently dequantized PyTorch reference
across 128-by-128 scale blocks.

On BW151, hipBLASLt reports that no suitable blockwise FP8 algorithm is
available for the reference shape. The export/signature test still runs, and
the numerical case is explicitly skipped on that backend rather than reading
an uninitialized hipBLASLt output.

Exact output is stored in
`docs/migration/logs/phase3-gemm-hipblaslt-w8a8-blockwise-gemm.log`.
