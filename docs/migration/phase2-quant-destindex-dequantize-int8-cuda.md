# Quant migration: `destindex_dequantize_int8_cuda`

The existing implementation is directly re-exported as
`lightop.quant.destindex_dequantize_int8_cuda`; the top-level API remains
unchanged.

Validation uses a `4 x 3 x 64` INT8 tensor and independently reconstructs the
kernel formula `(q + 128 - zero) * scale` in BF16. It also checks that the old
and category paths are the same function object.

```bash
python -m pytest -q test/test_quant_destindex_dequantize_int8_cuda.py
python tools/compat/generate_api_baseline.py --lmslim-root ../lmslim --check
```

Exact output is in
`docs/migration/logs/phase2-quant-destindex-dequantize-int8-cuda.log`.
