# Quant migration: `per_token_quant_fp8`

The function is directly exported from the source-preserving LMSlim module at
`lightop._lmslim_native.layers.gemm.fp8_utils`. Its name and signature are
unchanged.

The comparison test checks exact output parity with LMSlim for a `3 x 257`
FP16 input, covering a non-power-of-two hidden dimension.

```bash
PYTHONPATH="$PWD/build/lib.linux-x86_64-cpython-310:$PWD/../lmslim" \
  python -m pytest -q test/test_quant_per_token_quant_fp8.py
```

Exact output is in `docs/migration/logs/phase2-quant-per-token-quant-fp8.log`.
