# Quant migration: `per_token_quant_int8`

The original LMSlim `layers/gemm/int8_utils.py` file is retained under
`lightop._lmslim_native.layers.gemm`; only its environment import is changed
to the private relative path. `lightop.quant.per_token_quant_int8` directly
re-exports the migrated function.

The comparison test verifies the function signature and exact output parity
with LMSlim on a `3 x 257` BF16 input.

```bash
PYTHONPATH="$PWD/build/lib.linux-x86_64-cpython-310:$PWD/../lmslim" \
  python -m pytest -q test/test_quant_per_token_quant_int8.py
```

Exact output is in `docs/migration/logs/phase2-quant-per-token-quant-int8.log`.
