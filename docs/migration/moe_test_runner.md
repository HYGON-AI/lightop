# MoE 分类迁移测试清单

## 核心验收

构建完成后，在 LightOp 仓库根目录执行：

```bash
python test/run_moe_tests.py
```

如构建目录不在默认 `build/lib.*`：

```bash
python test/run_moe_tests.py --build-dir /path/to/build/lib.*
```

也可先触发完整构建，或只查看 runner 中的 case：

```bash
python test/run_moe_tests.py --build
python test/run_moe_tests.py --list-cases
```

核心 runner 覆盖：

- `lightop.moe` 公开 API 清单和旧顶层 API 的直接重导出关系。
- lmslim 实际依赖的 13 个 MoE Python API。
- MoE fused gate sqrt-softplus。
- fused GroupGEMM permutation + quant。
- Kimi K2 fused gate。
- `moe_align_block_size`。
- `moe_sum`。
- 通用 fused gate。

## lmslim 重点接口

lmslim 当前直接依赖：

- W8A8：`moe_gemm_w8a8`、`moe_gemm_marlin_w8a8`、
  `moe_gemm_marlin_w8a8_fp8` 及对应配置接口。
- W4A8：`moe_gemm_w4a8`、`moe_gemm_marlin_w4a8` 及对应配置接口。
- GroupGEMM W4A8：`moe_groupgemm_w4a8`、
  `moe_groupgemm_marlin_w4a8` 及对应配置接口。
- 原生扩展：`lightop.op.moe_align_block_size`、`lightop.op.moe_sum`。

对应 lmslim 调用入口位于：

- `../lmslim/lmslim/layers/fused_moe/`
- `../lmslim/lmslim/layers/groupgemm/groupgemm_w4a8.py`
- `../lmslim/lmslim/vllm_compat/moe_align_block_size.py`

## 扩展正确性脚本

以下脚本尺寸较大、需要额外参数或依赖，不放入核心 runner，按目标模型选择执行。

| 算子族 | 正确性脚本 |
| --- | --- |
| W8A8 | `test/moe_w8a8/fuse_moegemm_correctness.py` |
| W8A8 Marlin | `test/moe_w8a8/fuse_moegemm_correctness_marlin.py` |
| W4A8 | `test/moe_w4a8/fuse_moegemm_correctness.py` |
| W16A16 | `test/moe_w16a16/fuse_moegemm_w16a16_correctness.py` |
| W4A8 GroupGEMM | `test/moe_groupgemm_w4a8/w4a8_moe_groupgemm_correctness.py` |
| W4A8 GroupGEMM Marlin | `test/moe_groupgemm_w4a8/w4a8_moe_groupgemm_marlin_correctness.py` |
| token permutation/unpermutation | `test/test_groupgemm_permute_unpermute.py` |
| MoE SwiGLU dynamic quant | `test/test_moe_swiglu_dynamic_quant.py` |

GroupGEMM 示例：

```bash
python test/moe_groupgemm_w4a8/w4a8_moe_groupgemm_correctness.py \
  --E 16 --n1-list '[4096]' --n2-list '[7168]' --k-list '[2048]'

python test/moe_groupgemm_w4a8/w4a8_moe_groupgemm_marlin_correctness.py \
  --E 16 --n1-list '[4096]' --n2-list '[7168]' --k-list '[2048]'
```

## 已知环境要求

- `test/test_groupgemm_permute_unpermute.py` 依赖 `transformer_engine`。
- 部分性能脚本写 Excel，需要 `openpyxl`；数值测试可能已通过，但脚本仍会因
  写文件失败返回非零。
- GEMM/Marlin 脚本会按模型尺寸分配较大显存，应根据实际 EP/TP、专家数和
  hidden size 选择参数。

## 2026-07-16 当前验证结果

```text
PASS  pytest acceptance cases
PASS  test/test_kimi_k2_moe_fused_gate.py
PASS  test/test_moe_align_block_size.py
FAIL  test/test_moe_sum_opt.py
FAIL  test/test_fused_moe_gate.py
```

说明：

- `lightop.moe` API 验收共 9 项全部通过，包含 lmslim 所需 13 个 MoE
  Python API 的分类入口检查。
- lmslim 的 7 个 MoE/GroupGEMM 调用模块通过 `py_compile`。
- `moe_sum` 各测试尺寸的数值检查均通过；失败原因仅为环境缺少
  `openpyxl`，最后写 `output_reduce.xlsx` 时返回非零。
- 通用 `moe_fused_gate` 在既有正确性脚本中存在权重误差：
  `max_abs=2.7919e-01`、`max_rel=6.0000e-01`。
- Kimi K2 fused gate 在 batch size 1 到 8192 的全部测试尺寸通过。
- `moe_align_block_size` 的 50 组 token 数精度测试全部通过。
