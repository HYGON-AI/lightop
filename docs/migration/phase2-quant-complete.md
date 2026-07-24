# Phase 2 quant migration completion

The `lightop.quant` category now exposes all eight APIs in the approved
mapping:

- `blockwise_scaled_cast_transpose`
- `destindex_copy_quantize_int8_cuda`
- `destindex_dequantize_int8_cuda`
- `fp8_to_16`
- `w4tow16`
- `per_token_group_quant_fp8`
- `per_token_quant_fp8`
- `per_token_quant_int8`

The five existing LightOp implementations are direct category re-exports and
retain their top-level compatibility paths. The three LMSlim APIs are direct
exports from source-preserving modules under `lightop._lmslim_native`.

Existing operator scripts for blockwise scaling, WAN INT8 quantization,
`w4tow16`, and `fp8_to_16` now import the classified APIs. Focused pytest
tests cover API identity, native execution, explicit numerical references,
and LMSlim source parity.

The complete build and regression commands and their exact results are in
`docs/migration/logs/phase2-quant-complete.log`.
