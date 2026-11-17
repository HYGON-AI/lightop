---
name: lightop-kernel-knowledge
description: 在 HCU/ROCm 上新增或优化 LightOp 融合算子且需要 HIP 优化方法或实现证据时使用：检查本地源码、测试与配置，检索 ROCm/HIP/MIOpen/rocBLAS/hipBLASLt/Composable Kernel、Triton、PyTorch、SGLang 或 vLLM 中已验证的 HIP/ROCm 实现，查阅官方文档；内置 CUDA PR 语料仅用于跨平台启发。
allowed-tools: Bash Read Grep Glob
---

## 内置索引与证据边界

查询前阅读[索引与缓存说明](references/corpus-packaging.md)。从本 Skill 根目录运行内置 `scripts/`。本精简包仅含离线索引查询和按条目获取证据所需的脚本；批量抓取、clone、索引生成/校验工具由 hcu-infer-pilot 维护。查询不会联网；首次获取未缓存 diff 必须显式使用 `fetch_evidence.py --fetch`。

# LightOp 算子知识

本 Skill 将 LightOp/HCU 算子问题转化为可引用的实现证据。优先本地 LightOp 证据，其次是 ROCm/HCU 上游证据，最后才将跨平台 CUDA 证据用于启发。

适合回答以下问题：

- LightOp 如何暴露和测试相似算子？
- 新算子应放在哪个现有 `csrc/kernels/` 或专用家族中？
- 当前 kernel 由哪张配置表或哪个 dispatcher 分支选择？
- ROCm/PyTorch/Triton/SGLang/vLLM 是否已经提供相似的 HCU 路径？
- 下一次修改应依据哪个 HCU profiler 信号？
- 某个 CUDA PR 的思路能否迁移到 HIP/HCU，需要转换哪些内容？

## HIP 方法知识路线（优化前读取）

除本地源码证据外，方法选择优先参考同一安装目录下的 `kernel-hip-handbook` 与 `kernel-hip-optimize`。先定位这两个 Skill 的实际根目录，再按下表加载所需文件；不要把目录名当作已读证据。内置副本位于 agent-loop 的 assets 内，同级 Skill 应从外层 skills 目录定位，不能按 assets 的同级目录猜测。

| 需要解决的问题 | 来源中的文件 |
| --- | --- |
| roofline、优化空间与结构重构 | `kernel-hip-handbook/SKILL.md`、`references/launch-and-refactor.md` |
| 访存/LDS、资源压力与 wave 归约 | handbook 的 `references/memory-hierarchy.md`、`occupancy.md`、`wavefront-optimization.md` |
| shape 特化和 fallback | handbook 的 `references/shape-specialization.md` |
| 方法触发、跳过条件与兼容性 | optimizer 的 `references/optimization_catalog.md`、`method_registry.json` |
| PMC/SQTT 解释、目标 ISA 核验 | optimizer 的 `references/hcu_metrics_guide.md`、`hcu_isa_signatures.json` |
| 分支验证、消融和有效性状态 | optimizer 的 `references/optimizer-loop.md` |

查阅后将知识变为可检验假设：`来源及版本/哈希 → 适用 gfx/算子契约 → 当前测量证据 → 一项修改 → 预期指标变化 → 正确性/benchmark/ISA/消融验收 → 不适用或停止条件`。目录中的方法 ID 必须实际查到；找不到则使用描述性假设名。没有 HCU 实测的跨平台案例只标记为启发。

必须保留以下证据边界：

- 硬件规格、bank 数、occupancy 上限、寄存器公式和示例收益不能跨型号照搬；以目标设备、DTK、资源输出和实测带宽核对。缺失来源或工具要记录，不伪造数据。
- gfx936/gfx938 的 wave64、向量对齐与尾部、LDS layout、waitcnt/barrier 和 MMAC 操作数布局需要逐项验证；builtin 必须有精确源码签名、目标编译探针及最终 ISA。
- 低 occupancy 不自动等于瓶颈；提升 occupancy、使用 LDS、增加 unroll 或降低精度都需要触发证据和语义条件。
- 模板/自动生成器和复杂 kernel family 建议不代表实现已成立。LightOp 保持 `.cu`、原有绑定及构建/测试；独立 `solve(...)` 实验不能替代集成验证。
- 分开记录正确候选加速与方法贡献；消融须控制其它变量。完整相关 ISA 缺失、dump 不全、方法无收益和编译/正确性失败应分别记录，不能混成“优化无效”。
- 本知识流程不依赖 Humanize/RLCR。两份 HIP Skill 缺失时继续本地源码/内置证据检索，声明限制并使用已核实知识，不安装循环运行时来补位。

研究摘要除原有引用信息外，记录方法/假设、触发指标、资源预算、预期 ISA（适用时）、候选结果与反例。经验卡先作为待验证知识；未核验正确性、目标工作负载收益与归因前，不自动批准经验或重建外部 RAG 索引。

## 证据路线

可以使用一条路线，也可以组合多条。让证据影响代码前，先在研究摘要或尝试台账中记录使用的路线。

### 路线 A：本地 LightOp 源码

这是首选路线。定位包含以下内容的 LightOp checkout：

```text
setup.py
python/lightop/__init__.py
csrc/pybind/export.cpp
test/ci_tests/
```

这些路径对应重构后的 checkout；检索前核实所选版本布局。遵循主工作流的可选参数、文件复用、分类 API 和 `test/ci_tests/` 审查规则。提出新 `.cu` 或重复 wrapper 前，先检索已有实现。

先检索以下位置：

```bash
rg -n "<op_name>|<binding_name>|<kernel_name>" python/lightop csrc test setup.py
rg -n "m\.def\(\"<op_name>|<wrapper_name>" csrc/pybind/export.cpp python/lightop
rg -n "get_.*config|gfx|gcnArchName|PYTORCH_ROCM_ARCH" python/lightop/config.py python/lightop/config_deepgemm.py python/lightop/config_smooth_gemm.py python/lightop/moe/config.py python/lightop/tuning setup.py
```

需要记录的证据：

- Python wrapper 路径和公共 API 导入路径。
- `export.cpp` 的绑定行。
- `csrc/` 下的 kernel 实现文件。
- 配置或 dispatcher 分支。
- 测试与 benchmark 文件。
- 基线命令及结果。

### 路线 B：ROCm/HCU 上游与官方文档

HCU 专用实现或性能分析问题使用此路线。优先官方文档和上游源码，不优先博客或零散片段。

可用的来源类别：

- HYGON-AI/LightOp 公开源码参考：`sources/refs/hygon-lightop.md`，记录核验过的公开提交及源码路径。
- HYGON-AI/flash-attention-das 公开源码参考：`sources/refs/hygon-flash-attention-das.md`；源码入口与测试路径固定到已核验的公开提交。
- `sources/refs/` 下的源码参考页，尤其是：
  - `ref-hygon-hip-kernel-optimizer`：Hygon/HCU HIP 优化流程、方法分类、性能分析信号、wave64 规则、优先 CK Tile 的指导、基于源码的 builtin 使用规范，以及 `dccobjdump` ISA 核验。
  - HCU 部署与工程参考：`ref-hygon-inference-cookbook`、`ref-hygon-skillhub`。这些是已核验的公开来源入口。
- 框架 trace 范围归因参考 HYGON-AI/SkillHub 的 `torch-trace-operator-profiler`；DTK 原生计数器和命令含义以目标环境安装版本的工具帮助为准。
- ROCm/HIP 文档及 profiler 文档。
- PyTorch ROCm 实现与测试。
- MIOpen、rocBLAS、hipBLASLt、Composable Kernel、AITer。
- Triton 的目标平台后端。
- SGLang/vLLM 的目标平台路径与测试。

需要记录的证据：

- URL 或本地来源路径。
- 可获得的 commit SHA 或版本。
- 实际影响 LightOp 修改的文件、函数或配置。
- 复制或改编代码时的许可证和 notice 细节。

### 路线 C：内置 PR 语料

内置语料仍含有用的 kernel 工程案例，但大部分面向其他平台。仅在路线 A、B 证据不足，或需要跨平台算法思路时使用。

从知识根目录运行：

```bash
python3 scripts/query.py "<operator> <dtype> <symptom>" --compact --limit 30
python3 scripts/query.py "lightop hcu <operator>" --repo HYGON-AI/lightop --compact --limit 20
python3 scripts/query.py "flash attention hcu" --repo HYGON-AI/flash-attention-das --compact --limit 20
python3 scripts/query.py "hygon hcu mmac" --type source-reference --compact --limit 20
python3 scripts/query.py "hcu hip profiler" --type source-reference --compact --limit 20
python3 scripts/get_page.py <page-id>
python3 scripts/fetch_evidence.py <page-id>          # 只读取已有缓存或完整库
python3 scripts/fetch_evidence.py <page-id> --fetch  # 显式允许网络获取
```

借鉴思路前核验原始证据。`artifact_dir` 是完整库中的相对目录，不代表随索引提供了该文件。`fetch_evidence.py` 优先查本地完整库或缓存；自动网络获取仅支持 GitHub PR 的 diff，其他来源按页面链接通过有权限的工具获取。不会自动 clone 仓库或抓取所有源码。若已配置完整库，证据包通常包含：

```text
evidence/pull-bundles/<repo>/gh-<id>/review.diff
evidence/pull-bundles/<repo>/gh-<id>/source-snapshot/
evidence/pull-bundles/<repo>/gh-<id>/upstream.json
evidence/pull-bundles/<repo>/gh-<id>/ORIGIN.yaml
evidence/pull-bundles/<repo>/mr-<id>/review.diff
evidence/pull-bundles/<repo>/mr-<id>/source-snapshot/
evidence/pull-bundles/<repo>/mr-<id>/upstream.json
evidence/pull-bundles/<repo>/mr-<id>/ORIGIN.yaml
```

迁移规则：

- CUTLASS/CuTe/PTX/SASS/TMA/WGMMA/tcgen05/Nsight 证据在完成转换前均视为不可直接移植。
- 显式将 CUDA 概念映射到 ROCm：SM 对应 CU，适用时将 warp 对应到 wavefront，shared memory 对应 LDS，tensor-core 路径对应 MFMA 或选定的 ROCm 库/后端。
- Hygon/HCU 专属结论优先使用目标编译证据：`hipprof`、code-object 资源分析、可选 SQTT 和最终 `dccobjdump` 输出。在 Hygon ISA 显示 `v_mmac`、MMOP、matrix-load、DS matrix-read、direct-to-LDS 或其他预期 HCU 模式前，跨平台矩阵指令与张量核术语只作为类比。
- 只有精确签名存在于 HCU 来源页、现有工程源码或编译探针中时，才使用目标平台软件栈中有源码依据的内建函数。不能从指令助记符或其他平台的通用文档臆造 builtin 名称。
- 不以 CUDA profiler 指标证明 HCU 瓶颈。
- CUDA 来源应记录为启发，不能当成 HCU 实现已经成立的证据。

### 路线 D：补充源码与完整证据

需要完整源码或离线库时，按索引记录的 URL、版本和文件路径获取。只下载与当前任务相关的证据，并保存来源、版本和内容哈希。维护者可用 hcu-infer-pilot 的批量工具生成离线证据树；本包不提供一个已上线的全量库下载地址，也不自动下载约 1.1 GB 的历史资料。

`KERNEL_KNOWLEDGE_ROOT` 可指向已经准备好的完整知识目录（至少包含 `sources/`）。不能把索引命中或未取得的 diff 当作源码验证通过。

## 查询模式

LightOp 源码查询：

```text
rmsnorm export.cpp wrapper test config
moe w8a8 gfx936 config benchmark
fuse silu quant op binding csrc
HYGON-AI/lightop gfx938 fused op test
HYGON-AI/flash-attention-das hcu attention benchmark
hygon hcu mmac wave64 lds waitcnt
HYGON-AI HCU deployment profiler source
```

HCU 性能分析查询：

```text
hipprof launch overhead LightOp <op>
rocprof-compute LDS bank conflict HIP kernel
ROCm VGPR occupancy HIP kernel
MFMA utilization Composable Kernel grouped gemm
```

跨平台启发查询：

```text
rmsnorm vectorized load reduction PR
moe grouped gemm quantization dispatch PR
topk softmax fusion benchmark PR
```

## 引用检查清单

每项发现采用相同的引用结构：

- 路线名称。
- 来源路径或 URL。
- 可获得的提交或版本。
- 关键函数、绑定、配置键、测试或 benchmark。
- 这些证据改变了 LightOp 计划的什么内容。
- 思路是否能直接移植到 HCU，还是仅用于跨平台启发。
- 复制或改编代码时的许可证和 notice 细节。

某条路线匹配很少或为空时，先扩大查询词或交叉检查另一条路线，再将其当作发现。

## 验证

本地索引查询与缓存读取仅验证资料可读，不证明算子正确。索引重建与全库验证属于 hcu-infer-pilot 的维护流程。验证 LightOp 方案仍需执行主工作流要求的目标构建、独立数值测试、同卡 benchmark 与所需 profiling；未取得原始证据时明确标为缺失。
