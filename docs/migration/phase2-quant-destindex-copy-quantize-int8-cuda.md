# Quant migration: `destindex_copy_quantize_int8_cuda`

`lightop.quant.destindex_copy_quantize_int8_cuda` is a direct re-export of
the existing top-level API. The compatibility path and native PyBind entry
remain unchanged.

Validation on Python 3.10.12, PyTorch 2.10.0, HIP 6.3.26113, and BW151:

```bash
python -m pytest -q test/test_quant_destindex_copy_quantize_int8_cuda.py
python test/test_quantization_cuda.py
python tools/compat/generate_api_baseline.py --lmslim-root ../lmslim --check
```

The focused test checks direct symbol identity and executes the category API
on a `4 x 3 x 64` BF16 tensor. The existing Triton comparison test covers
quantized values, scale/zero values, dequantization, and benchmark launches.

Exact command output is stored in
`docs/migration/logs/phase2-quant-destindex-copy-quantize-int8-cuda.log`.
