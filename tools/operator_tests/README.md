# Isolated operator test runs

The runner uses the LightOp package already installed in the current image.

Post-integration image:

```bash
python tools/run_operator_tests.py
```

Baseline image:

```bash
python tools/run_operator_tests.py --stage before
```

Each run writes complete per-case logs, `passed_cases.txt`,
`failed_cases.txt`, `summary.txt`, and `summary.json` under
`test_results/<stage>_<timestamp>/`. The terminal also prints each case and
the final totals.

To test a wheel without changing the current environment, `--package` remains
available and installs it into the result directory:

```bash
python tools/run_operator_tests.py --package /path/to/lightop.whl
```
