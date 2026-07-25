# norm / activation 全 case 验收脚本

构建完成后，在 LightOp 仓库根目录执行：

```bash
python test/run_norm_tests.py
python test/run_activation_tests.py
```

如构建目录不在默认 `build/lib.*`：

```bash
python test/run_norm_tests.py --build-dir /path/to/build/lib.*
python test/run_activation_tests.py --build-dir /path/to/build/lib.*
```

也可由脚本先完整构建：

```bash
python test/run_norm_tests.py --build
python test/run_activation_tests.py --build
```

覆盖范围：

- `norm`：22 个既有 API 直接重导出、完整公开清单、LMSlim RMSQuant 绑定/签名、三组 hidden size 数值、scale、input/residual 原地更新。
- `activation`：19 个既有 API 直接重导出、完整公开清单、LMSlim SiLU-mul quant 绑定/签名、FP16/BF16、三组 shape、INT8 输出和 scale。

脚本强制检查 `build/lib.*/lightop/op*.so`，防止构建产物缺失时回退到系统安装包。任一 case 失败，脚本返回非零退出码。
