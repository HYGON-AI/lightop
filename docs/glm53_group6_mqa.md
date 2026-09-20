# GLM5.3 group6 原生 FP8 MQA 调优

group6 路径允许每个 CTA 复用同一请求的 2、3 或 6 行 Q，保持原来的
4-wave FP8 MMAC 和 persistent 调度。默认仍复用2行，通过环境变量选3行：

```bash
export LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA=3
```

该变量也接受 `2`、`6`；其他值会报错。组1/3/4/5的默认行为保持原样。
在 gfx938、64/72 CU、原生 FP8、group6、总共30行的 BS5 verify 场景，Python
入口将自动 CTA 数量设为128。调用者显式传入的非零 CTA 数量保持原样。

此优化集成在常规 `lightop.op` 中，不需要额外加载调优用的独立扩展。
构建命令与原分支一致：

```bash
PYTORCH_ROCM_ARCH=gfx938 python3 setup.py bdist_wheel
```

GLM5.3、EP32、单 DP BS5、MTP5/1/6 的实测配置使用3行复用和128 CTAs。
原有微基准约334.2微秒降到290.2微秒，有效 logits 与 sparse mask 逐位一致，
并覆盖空行、页边界和图重放。该数字只对应 MQA 微基准，不代表整机吞吐提升。

稀疏 MQA 仍需与 mask-aware TopK 消费端成对使用。本提交没有修改
TopK-transform 的算法或选择路径。
