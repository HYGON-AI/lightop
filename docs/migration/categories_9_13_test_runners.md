# Categories 9-13 acceptance runners

The migration acceptance cases are grouped by their new public namespaces:

| Category | Namespace | Runner |
| --- | --- | --- |
| 9 | `lightop.kvcache` | `test/run_kvcache_tests.py` |
| 10 | `lightop.sampling` | `test/run_sampling_tests.py` |
| 11 | `lightop.tensor` | `test/run_tensor_tests.py` |
| 12 | `lightop.vision` | `test/run_vision_tests.py` |
| 13 | `lightop.loss` | `test/run_loss_tests.py` |

Run the full final acceptance suite from the repository root:

```bash
python test/run_categories_9_13_tests.py --build
```

Reuse an existing build:

```bash
python test/run_categories_9_13_tests.py
```

Inspect all included cases without running them:

```bash
python test/run_categories_9_13_tests.py --list-cases
```

A single category can be accepted independently with its runner, for example:

```bash
python test/run_kvcache_tests.py --build
```

The runners prepend the selected `build/lib.*` directory to `PYTHONPATH`.
They stop with a non-zero exit status when a build, pytest file, or standalone
legacy test script fails.
