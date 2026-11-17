# HCU 性能分析指标

编写 `hcu-profiler-report` 摘要时使用以下分组。优先采用已安装 DTK/ROCm profiler 输出中的准确指标名；本文列出应关注的信号，不是跨版本固定 schema。

## 耗时

- 显式同步后的 benchmark p50/p90/mean 时延。
- `hipprof` 总运行时间、HIP API 时间、kernel 时间、memcpy/memset 时间。
- 最热点 kernel 的持续时间及其占采集运行时间的比例。
- 大量微小 kernel 主导时的 launch 次数和启动开销。

## 环境

- 通过 `torch.cuda.get_device_properties(0).gcnArchName` 获取 HCU 型号和 gfx 架构。
- DTK/ROCm 版本、PyTorch HIP 版本、`hipcc --version`。
- `PYTORCH_ROCM_ARCH`、构建命令、LightOp commit 和生效配置分支。

## 内存路径

- 全局 load/store 流量与吞吐。
- 热点 load/store 的对齐和向量宽度。
- 可用的 L2/HBM 吞吐或利用率计数器。
- `hipprof` 中主机与设备 memcpy/memset 时间。
- benchmark 代码可见的临时张量分配或物化。

解释：

- memcpy/memset 时间高，指向数据搬运或分配，不是 kernel 数学运算。
- 全局流量高而计算利用率低，提示应考虑融合、向量化访存、布局或冗余读写修复。
- 小 shape 伴随大量 launch，通常更需要融合或 dispatch 修改，而非微 kernel 调优。

## LDS 与同步

- 每 block 的 LDS 用量，以及可用的 LDS bank-conflict 计数器。
- 可用的 barrier/wait 计数器或时间线空隙。
- 通过源码/ISA 检查得到的暂存缓冲大小和访问 stride。

解释：

- bank conflict 提示应考虑 padding、swizzle 或不同的逐 wave 布局。
- barrier 很重的 kernel 可能需要减少阶段、简化暂存，或改变 wave/block 间的分工。

## 计算路径

- GEMM/MoE 路径的 MFMA 利用率或指令数。
- 逐元素与归约 kernel 的 VALU/SALU 指令组成。
- int8/fp8/w4 路径的转换/量化指令数。
- epilogue 开销，尤其是 bias/activation/quant/store 融合。

解释：

- 类 GEMM 路径的 MFMA 信号低，通常意味着没有选择预期 kernel 路线或配置。
- 简单逐元素 kernel 的标量/控制开销高，可能来自分支繁多的 dispatch、热循环中的 shape 检查或标量化代码生成。

## 资源

- VGPR/SGPR 数量。
- Scratch/spill load/store。
- 每 block 的 LDS 字节数。
- profiler 提供时的实际 occupancy 或其限制项。
- block 大小、每 block 的 wave 数、grid 大小和 CU 数量。

解释：

- VGPR 压力和 spill 会让看似更好的融合 kernel 变慢。
- 相对 CU 数量，block 太少会导致 occupancy 不足或尾部 wave。
- 微小 block 过多会放大 launch/config 开销。

## Dispatch 与 shape 覆盖

- 从当前算子实际使用的配置模块（如 `python/lightop/config.py`、`python/lightop/moe/config.py` 或 `python/lightop/tuning/`）选中的配置 key。
- 选中的 gfx 专用分支。
- fallback 路径与预期优化路径。
- `W` 中各 shape 的 benchmark 变化量。

解释：

- 仅一个区间改善时，将候选限制在 dispatcher/config 保护条件后，不直接替换默认路径。
- 错误配置被命中时，先修配置表，再改 kernel 数学运算。

## ISA / Code Object

profiler 指向代码生成问题时，检查目标 ISA 或 code-object 元数据：

- 热点指令窗口。
- 内存操作的向量宽度。
- MFMA 指令形状。
- scratch 内存访问。
- 过度标量化或 predicate/control flow。
- 内联/展开压力。

根据 ISA 证据选择小范围编辑：launch bounds、向量类型、展开因子、拆分 kernel、配置特化或源码层布局改写。
