# Phase 3: existing LightOp GEMM APIs

The 23 existing LightOp GEMM and Grouped GEMM APIs are directly re-exported
through `lightop.gemm_ops`. Their top-level compatibility paths and native
implementations are unchanged.

The public package is named `gemm_ops` so importing it cannot replace the
callable `lightop.gemm` attribute with a module. APIs are split internally
into `core.py`, `quantized.py`, and `grouped.py`; these modules contain only
direct symbol bindings.

Validation checks every exported function by object identity and verifies
that `lightop.gemm` remains callable after importing `lightop.gemm_ops`.
Numerical and performance coverage continues to use the existing operator
tests and is rerun after all six LMSlim GEMM APIs are migrated.

Exact output is stored in
`docs/migration/logs/phase3-gemm-ops-existing-apis.log`.
