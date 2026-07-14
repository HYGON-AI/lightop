# Quant migration: `per_token_group_quant_fp8`

The LMSlim `layers/gemm/fp8_utils.py` file is copied into
`lightop._lmslim_native.layers.gemm` without splitting its kernels or helper
functions. Its only project import is redirected from `lmslim.envs` to the
private LightOp copy. The public API is a direct re-export at
`lightop.quant.per_token_group_quant_fp8`.

The comparison test checks the complete function signature and exact output
parity with the source LMSlim function for a `4 x 256` BF16 input and group
size 128.

```bash
PYTHONPATH="$PWD/build/lib.linux-x86_64-cpython-310:$PWD/../lmslim" \
  python -m pytest -q test/test_quant_per_token_group_quant_fp8.py
```

Exact output is in
`docs/migration/logs/phase2-quant-per-token-group-quant-fp8.log`.
