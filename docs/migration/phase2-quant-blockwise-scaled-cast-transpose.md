# Phase 2: `blockwise_scaled_cast_transpose`

## Scope

This change migrates the existing LightOp operator into the public
`lightop.quant` category without moving or rewriting its implementation.

- New category API: `lightop.quant.blockwise_scaled_cast_transpose`
- Preserved compatibility API: `lightop.blockwise_scaled_cast_transpose`
- Implementation: `lightop.blockwise_scaling.blockwise_scaled_cast_transpose`
- Native entry point: `lightop.op.blockwise_scaled_cast_transpose`

The category API is a direct symbol re-export. Both public paths therefore
refer to the same Python function object and use the same C++/HIP kernel.

## Files

- `python/lightop/quant/blockwise.py`: direct category-layer re-export.
- `python/lightop/quant/__init__.py`: explicit public export and `__all__`.
- `test/test_quant_blockwise_scaled_cast_transpose_api.py`: category and
  compatibility identity checks.
- `test/test_blockwise_scaled_1d_cast_transpose.py`: unchanged numerical
  reference regression for the shared implementation.

## Validation

Environment: Python 3.10.12, PyTorch 2.10.0, HIP 6.3.26113, BW151 GPU.

The compiled extension was loaded from
`build/lib.linux-x86_64-cpython-310`, while the changed Python files were
copied into that build tree before validation.

```bash
export PYTHONPATH="$PWD/build/lib.linux-x86_64-cpython-310"
python -m pytest -q test/test_quant_blockwise_scaled_cast_transpose_api.py
python test/test_blockwise_scaled_1d_cast_transpose.py
python tools/compat/generate_api_baseline.py --lmslim-root ../lmslim --check
python -m compileall -q \
  python/lightop/quant \
  test/test_quant_blockwise_scaled_cast_transpose_api.py
```

Results:

- Category/compatibility API test: `1 passed in 2.11s`.
- Numerical reference test: passed for a `2048 x 2000` float32 input,
  compact row/column scales, transpose enabled, INT8 output, epsilon `0`,
  and power-of-two scales. Reported kernel average: `0.13884100914001465 ms`.
- Compatibility manifests: 143 Python APIs, 209 native symbols, and 22
  LMSlim-required APIs unchanged.
- Python bytecode compilation: passed.

The timing is recorded only as test context; it is not treated as a
performance comparison because this migration does not change the kernel.
