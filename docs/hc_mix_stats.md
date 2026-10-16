# hc_mix_stats：混合精度 GEMM 与 RMS 统计融合

## 分类与入口

`lightop.gemm_ops.hc_mix_stats` 属于 **GEMM 算子族 / 混合精度 GEMM 融合算子**，同时通过 `lightop.hc_mix_stats` 导出。唯一公开接口为：

```python
hc_mix_stats(x, weight, eps=1e-6) -> Optional[Tensor]
```

输入为连续 BF16 `x[M,K]`、连续 FP32 `weight[N,K]` 和正有限 epsilon，支持时输出为 FP32 `[M,N]`，不支持时返回 `None`。C++ 内部顺序执行两个原生 kernel：在线权重拆分、GEMM 与平方和部分归约；随后汇总并应用 RMS 缩放。中间量和 workspace 不对外暴露，调用方无需传入原始 K 或分片数。Sinkhorn 仍由 SGLang 原有路径执行。

```python
import torch
from lightop.gemm_ops import hc_mix_stats

x = torch.randn(17, 257, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(33, 257, device="cuda", dtype=torch.float32)
mixes = hc_mix_stats(x, weight, eps=1e-6)  # Tensor 或 None
reference = (x.float() @ weight.T) * torch.rsqrt(
    x.float().square().mean(-1, keepdim=True) + 1e-6
)
if mixes is None:
    mixes = reference
torch.testing.assert_close(mixes, reference, atol=1e-4, rtol=1e-4)
```

## 数学语义与形状

输入 `x[M,K]` 为 BF16，`weight[N,K]` 为 FP32。每次调用在 GPU 内计算：

```text
wh = BF16(weight)
wl = BF16((weight - FP32(wh)) * 256)
S  = ceil(K / 256)
Is = [256*s, min(256*(s+1), K))
hi[s] = FP32_GEMM(x[:,Is], wh[:,Is].T)
lo[s] = FP32_GEMM(x[:,Is], wl[:,Is].T)
sq[s] = sum(FP32(x[:,Is])**2, axis=1)
y = (sum(hi,axis=0) + sum(lo,axis=0)/256)
    * rsqrt(sum(sq,axis=0)[:,None]/K + eps)
```

第一阶段在内部生成三个连续 FP32 临时张量：`[S,M,N]`、`[S,M,N]`、`[S,M]`。第二阶段生成接口返回的连续 FP32 `[M,N]`。权重拆分的 high/low 是 BF16，乘加、平方和与最终结果是 FP32。high/low 表示是近似计算，并非任意 FP32 数值的精确分解。

`K=20480,N=24` 是 DeepSeek V4.1 Flash 的性能特化分支，不是接口限制。`256` 是分片长度，`S` 由 K 计算，因此原来的 80 是 `20480/256` 的结果。尾片补零，平方和分母仍使用原始 K。分片边界与逐行算术顺序不随 M 改变。

## 接口契约

| 项目 | 约束及行为 |
|---|---|
| 维度 | M、N 可为 0；K 必须大于 0；K/N 可非整 tile |
| dtype | 输入 BF16、权重 FP32；部分和和输出 FP32。其他 dtype 返回 None |
| 布局 | 二维连续输入；支持非对齐的连续 storage-offset 视图；不隐式 `.contiguous()` 或转 dtype |
| 设备 | 所有张量同一 gfx928、gfx936 或 gfx938 GPU；其他架构返回 None |
| 推理 | 仅用于推理，在 no_grad/inference_mode 下调用；无反向实现，不检查梯度开关，也不根据 requires_grad 触发回退 |
| 视图 | 不接受尚未 resolve 的 negative view bit；BF16/FP32 为实数类型 |
| 数值 | eps 必须为正、有限且转 FP32 后仍非零；输入应有限，high/low 可表示且 FP32 累加不溢出。NaN/Inf 不做清洗 |
| 内部第二阶段 | 三份部分和及原始 K 由公开接口内部传递；调用方无需传入或校验中间量 |
| 元数据边界 | M≤2³¹−1、K≤65535×256、N≤65535×24，受显存和 GPU grid 上限约束 |
| 原地修改 | 不修改任何输入；结果不与输入共享可写存储 |
| stream/graph | 使用当前 stream；支持 graph capture/replay 并更新 x/weight 内容 |
| 临时存储 | Torch allocator 管理，每次调用独立；无跨调用共享的可写 workspace，无浮点原子 |

返回状态由原生 C++ 的元数据预检统一决定，不在 Python/框架侧重复校验。不满足设备、架构、dtype、维度/形状、布局/view、epsilon 或 launch 上限时，在分配输出/临时缓冲区、启动 kernel 前返回 `None`。支持的空 M/N 返回空 Tensor，用 `is None` 判断状态，不能按 Tensor 的布尔值判断。设备查询故障、OOM、kernel 执行错误不转成 None，也不捕获为静默回退；Python 参数类型错误仍按绑定接口报错。原始输入若对 Torch 也不合法，回退后 Torch 仍可能报错，None 不会修复输入。

输入范围不是无条件的误差保证。测试使用 Torch FP32 原公式和逐元素 `1e-4 + 1e-4*abs(reference)` 容差，覆盖模型权重、常规/微小/宽动态范围/零输入、通用尾块和真实激活。极端溢出、严重病态相消等情况不在该测试误差保证内。

中间部分和由同一次调用产生并在内部传给归约 kernel，存储量为 `4*S*M*(2*N+1)` 字节。M 或 N 为 0 时直接返回空输出，不分配部分和、不启动 kernel。

## 按 token 数选择 tile

在 DeepSeek 目标 `K=20480,N=24` 且权重向量读取对齐时：

| 本次调用的 M | M tile | 物理 K wave 分组 | LDS |
|---|---:|---:|---:|
| M<256 | 16 | 4 | 12480 B |
| M≥256 | 32 | 2 | 8320 B |

M 是本 rank 本次 forward 的 token 行数：decode 通常对应本地活跃序列数，prefill 对应当前分片/chunk 的 token 数。当前模型 `hc_mult=4, hidden_size=5120`，所以 K=4×5120=20480，N=4²+2×4=24。

大 M 路径复用权重读取和在线拆分，两个物理 K 组分别完成逻辑分组(0,2)和(1,3)，然后相加。仍然计算同一个 `(h0+h2)+(h1+h3)` 树，因此跨 M 分发不改变逐行运算顺序。非目标形状或非对齐权重使用通用16行路径。

64行tile已实测并拒绝：扩大tile导致本机资源代价超过复用收益。这里的分界来自 BW1200/gfx938 实测，不代表其他架构最优。通用形状保证接口与测试覆盖，不承诺所有 N/K 都快于 Torch；本次性能优化重点是 N=24、K=20480、变化的 M。

## 实现与架构

在 HCU 上使用本地 MMAC、向量化读写及 wave shuffle。gfx928/936 对应 `__builtin_amdgcn_mmac_f32_16x16x16bf16`，gfx938 对应 `__builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts`，不能假定不同 gfx 的 builtin 签名相同。三种架构均开放接口并由构建配置编译对应设备代码；当前运行测试和性能调优在 gfx938 完成，gfx928/936 尚未进行实卡精度与性能验证。

参考 [Tencent hpc-ops 的 high/low 表示](https://github.com/Tencent/hpc-ops/blob/main/hpc/gemm.py) 和 [SM90 GEMM 的 tile/数据复用方案](https://github.com/Tencent/hpc-ops/blob/main/src/gemm/sm90/gemm.cu)。这里读取原始 FP32 权重并在线拆分，不使用腾讯接口要求的预拆分权重，也不移植 NVIDIA 专用 TMA/WGMMA 或跨 CTA 完成标志。

## 测试与 SGLang

正式测试入口为 `test/ci_tests/test_hc_mix_stats.py`。CI 正确性运行 `python -m pytest -v test/ci_tests/test_hc_mix_stats.py`；保留直接执行脚本的精度、模型权重和性能测试选项。运行 `--help` 查看完整选项；`--general-only` 聚焦通用形状，`--sglang-check` 验证框架的 LightOp/Torch 分发。`--triton-path` 仅用于外部历史基准对比，不是 SGLang 后端。

SGLang 复用全局开关 `SGLANG_USE_LIGHTOP`：HCU 且该开关为 `1` 时，调用统一的 `hc_mix_stats` 接口；关闭时走框架原有的 Torch 分支，NVIDIA 路径保持不变。没有独立的 hc_mix_stats 后端开关或 auto 模式。接口在模块加载时导入，早于 graph capture；开关需在启动进程前设置。启用后要求安装包含 `hc_mix_stats` 接口的 LightOp，当前原生 kernel 支持 gfx928/gfx936/gfx938；算子返回 None 时复用框架已有的 Torch 公式。扩展导入失败、设备查询故障、执行错误和 OOM 仍报错。dtype、形状、布局等校验由原生接口统一负责。框架没有新增 Triton kernel 或 Triton 后端。全局开关还会影响其他 LightOp 算子，不能把切换它的端到端差异单独归因于 hc_mix_stats。

SGLang 先调用 `mixes = lightop_hc_mix_stats(x_flat, hc_fn, self.rms_norm_eps)`；当 `mixes is None` 时执行已有 Torch 公式，最后统一 `unsqueeze(1)`。导入别名用于区分已有 CUDA 实现。性能基准统计整个接口；两阶段的 device 分项由 profiler 中的 kernel 事件观察，不再公开阶段接口。
