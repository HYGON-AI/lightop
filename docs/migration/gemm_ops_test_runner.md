# gemm_ops 统一测试脚本

脚本：`test/run_gemm_ops_tests.py`

在 LightOp 仓库根目录执行：

```bash
python test/run_gemm_ops_tests.py --build
```

测试阶段：

1. `gemm_ops` 既有接口、LMSlim 原生接口及重复注册兼容测试，其中直接包含 hipBLASLt blockwise FP8 数值测试。
2. W4A16 既有回归，严格检查 20 组输出均为 `test pass`。
3. W8A16 FP8 既有回归，严格检查 10 组数值比较均为 `True`。

FP8 不使用额外开关。在支持 FP8/INT8 的后端会与其他测试一起执行；硬件不支持时，由对应 pytest 用例自身报告 skip。

可选参数：

- `--skip-legacy`：仅运行 `gemm_ops` 迁移专项，跳过较长的既有回归。
- `--build-dir PATH`：指定要验证的 `build/lib.*` 目录。
- `--max-jobs N`：设置构建并发数，默认 16。

任一阶段失败时脚本最终返回非零退出码。
