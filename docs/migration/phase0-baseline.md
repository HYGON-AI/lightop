# Phase 0 Baseline Validation

## Scope

- Freeze the current LightOp Python exports.
- Freeze the current `lightop.op` pybind symbols.
- Freeze the LightOp API referenced by the unmodified lmslim package.
- Exclude `lightop/csrc/allreduce_fuse/`.

## Environment

```text
Python 3.10.12
PyTorch 2.10.0
HIP 6.3.26113
hipcc /opt/dtk/bin/hipcc
GPU available: true
```

## Validation

```bash
python tools/compat/generate_api_baseline.py --lmslim-root ../lmslim
python tools/compat/generate_api_baseline.py --lmslim-root ../lmslim --check
```

The committed manifests are the compatibility baseline for subsequent phases.
Any intentional API removal must update both the manifest and the migration
documentation in the same commit.
