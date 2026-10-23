# LightOp

LightOp 是面向海光 HCU 的高性能算子库，为 PyTorch 模型的推理和训练提供融合算子与低精度计算能力，覆盖 GEMM、MoE、Attention、归一化、量化等常用计算路径。

## 主要算子

| 类别 | 主要能力 | 接口入口 |
| --- | --- | --- |
| GEMM | 通用矩阵乘、Grouped GEMM、W8A8 / W4A8 / W4A16 量化计算 | [gemm_ops](python/lightop/gemm_ops/) |
| MoE | 专家路由、融合专家计算、数据重排与专家并行辅助算子 | [moe](python/lightop/moe/) |
| Attention / KV Cache | MQA Logits、分页与稀疏索引、RoPE、KV Cache 读写 | [attention](python/lightop/attention/)、[kvcache](python/lightop/kvcache/) |
| 归一化 | RMSNorm、LayerNorm、GroupNorm 及残差融合 | [norm](python/lightop/norm/) |
| 激活与量化 | SiLU / SwiGLU 融合、INT8 / FP8 量化与反量化 | [activation](python/lightop/activation/)、[quant](python/lightop/quant/) |
| 采样 | Top-K / Top-P 等采样辅助算子 | [sampling](python/lightop/sampling/) |
| 视觉与损失函数 | Grid Sample、上采样、CTC / RNNT Loss | [vision](python/lightop/vision/)、[loss](python/lightop/loss/) |

各算子支持的数据类型、输入形状和硬件架构以接口及对应测试为准。

## 安装

请在可访问海光 HCU 的 Linux 环境中编译，预先安装 DTK 及配套的 HIP 版 PyTorch、`setuptools`、`wheel` 和 `ninja`。构建环境可参考 [CI 配置](.github/workflows/pr-test.yml)。

在仓库根目录执行：

```bash
source /opt/dtk/env.sh
python3 setup.py install
```

如需生成 wheel 安装包：

```bash
python3 setup.py bdist_wheel
```

产物位于 `dist/`。可通过 `MAX_JOBS` 控制编译并行度，例如 `MAX_JOBS=32 python3 setup.py install`。

## 使用示例

通过归一化分类 `lightop.norm` 调用 RMSNorm；HIP 版 PyTorch 同样使用 `device="cuda"` 访问 HCU：

```python
import torch
import lightop.norm

x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device=x.device, dtype=x.dtype)
y = lightop.norm.rms_norm(x, weight, 1e-6, out=torch.empty_like(x))
print(y.shape)  # torch.Size([4, 4096])
```

更多调用方式见 [test/](test/)。

## 测试

安装好 LightOp 及 [CI 所列测试依赖](.github/workflows/pr-test.yml) 后，在仓库根目录执行：

```bash
cd test/ci_tests
python3 -m pytest -v -p no:cacheprovider --tb=short
```

性能结果受硬件、数据类型和输入形状影响，请以对应环境的实测结果为准。

## 许可证

LightOp 使用 [Apache License 2.0](LICENSE)。第三方组件遵循各自的版权与许可证，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
