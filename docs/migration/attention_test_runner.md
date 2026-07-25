# lightop.attention migration acceptance runner

The category runner collects the API compatibility check and all Attention, RoPE,
MQA and QKV functional scripts touched by this migration.

List the cases without running them:

```bash
python test/run_attention_tests.py --list-cases
```

Run against the latest existing `build/lib.*` package:

```bash
python test/run_attention_tests.py
```

Rebuild LightOp first, then run the complete acceptance suite:

```bash
python test/run_attention_tests.py --build --max-jobs 16
```

To validate a specific build directory:

```bash
python test/run_attention_tests.py \
  --build-dir build/lib.linux-x86_64-cpython-310
```

The full suite intentionally includes legacy performance-oriented scripts and may
take substantially longer than the per-operator smoke and correctness checks.
