# LightOp Migration Validation

This directory records reproducible validation results for the lmslim migration.

Compatibility manifests are generated with:

```bash
python tools/compat/generate_api_baseline.py \
  --lmslim-root ../lmslim
```

CI and local validation should use the same command with `--check` after the
manifests have been generated. The generator intentionally excludes
`lightop/csrc/allreduce_fuse/`, which is outside the maintained API surface.
