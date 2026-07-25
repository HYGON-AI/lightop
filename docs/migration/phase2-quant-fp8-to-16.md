# Quant migration: `fp8_to_16`

`lightop.quant.fp8_to_16` directly re-exports the existing implementation;
`lightop.fp8_to_16` remains unchanged.

The focused test uses a `256 x 256` E4M3 FP8 tensor and four distinct scales
to cover all 128-by-128 scale blocks. The output is compared exactly with a
PyTorch block-scale reference.

```bash
python -m pytest -q test/test_quant_fp8_to_16.py
python tools/compat/generate_api_baseline.py --lmslim-root ../lmslim --check
```

Exact output is in `docs/migration/logs/phase2-quant-fp8-to-16.log`.
