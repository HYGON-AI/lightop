# Quant migration: `w4tow16`

`lightop.quant.w4tow16` directly re-exports the existing packed-W4 to FP16
conversion while retaining `lightop.w4tow16`.

The focused test packs known nibble sequences into two UINT32 rows, applies
different FP16 scale/zero pairs, and compares all unpacked values exactly.
The existing `test/ci_tests/test_gemm.py` benchmark now imports `w4tow16` from
`lightop.quant`, so its original GEMM preparation path also exercises the
classified API.

```bash
python -m pytest -q test/test_quant_w4tow16.py
python tools/compat/generate_api_baseline.py --lmslim-root ../lmslim --check
```

Exact output is in `docs/migration/logs/phase2-quant-w4tow16.log`.
