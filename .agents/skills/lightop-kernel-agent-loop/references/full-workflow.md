# LightOp 算子智能体闭环

用户希望为 HCU/ROCm 新增 LightOp 算子或优化已有算子时，使用此流程。由 agent 直接执行 HIP 证据驱动闭环；优化选法、目标 ISA 核验和消融遵循 [HIP 优化规则](hip-optimization-rules.md)。

本流程面向 LightOp 的目标平台实现。优先考虑 HIP C++、ROCm PyTorch 扩展、已有 LightOp 模式、MIOpen/rocBLAS/hipBLASLt、Composable Kernel 或 Triton 的 HIP/ROCm 路径；所选路线必须已被本地仓库支持，或由用户明确要求。

## 输入契约

实施前恢复或定义以下内容：

```text
K: 算子语义、输入/输出张量、dtype/layout/mode 约束
R: 正确性参考，通常为原生 PyTorch、已有 LightOp 路径，或处理边界情况的小型显式 oracle
W: 工作负载分布，包括目标 shape、dtype、模型场景、gfx 架构、layout/连续性、
   reduction/axis 细节、epsilon 或 mode 参数、有效带宽公式，以及时延/吞吐比较目标
E: 执行环境，包括 Docker 容器/镜像或直接宿主、环境中的 LightOp 路径、可见 HCU 设备、
   DTK/ROCm/PyTorch 版本、构建命令、测试命令、benchmark 命令和通过阈值
```

仅当 `K`、`R`、`W`、目标 HCU/gfx 架构、比较基线、执行环境、通过阈值或硬性范围约束缺失，且无法从 LightOp 仓库安全推断时，才询问用户。输入齐备后，由闭环负责检查、研究、实现路线、性能分析深度、benchmark 扩展和调优谱系等战术选择。

## 定位 LightOp

默认在 LightOp checkout 上工作，不使用独立的 CUDA 实验。先读取适用的 `AGENTS.md` 并定位 Git 根目录。重构后的常见布局如下，实际以所选版本和构建配置为准：

```text
setup.py
python/lightop/__init__.py
csrc/pybind/export.cpp
test/ci_tests/
```

按以下顺序查找：

1. 用户提供的路径。
2. 当前工作目录。
3. 当前工作区附近名为 `lightop` 的同级目录。

没有找到 checkout 时，询问其路径。用户明确要求独立实验时，将实验放在 LightOp 外部，但记录最终候选如何合入 LightOp 源码树。

## 执行环境

构建、正确性、benchmark 和性能分析使用同一套环境。用户指定 Docker 容器或镜像时，将 Docker 纳入验收契约。优先使用非交互命令，便于捕获日志和复现同一条命令：

```bash
docker exec <container> bash -lc 'cd /path/in/container/lightop && <command>'
```

指定 Docker 为执行环境时，构建和验证使用同一容器及其源码路径。检查当前 `setup.py`、版本分支和实际构建命令，以仓库现有入口为准。安装包与服务共享时，使用隔离构建/包验证；编辑或构建源码不授权替换服务包。保留可用的增量构建产物。

用户提供镜像而非运行中的容器时，询问或推断宿主 LightOp 路径，并暴露 HCU 设备运行：

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --network=host \
  -v <host-lightop>:/workspace/lightop <image> \
  bash -lc 'cd /workspace/lightop && <command>'
```

首次构建前记录：

- 容器名或镜像 tag；不用 Docker 时记为 `direct-host`。
- 从执行命令视角看到的 LightOp 路径，不只记录宿主路径。
- `HIP_VISIBLE_DEVICES`、`HSA_VISIBLE_DEVICES` 等设备可见性环境变量。
- 性能运行前的 HCU 状态命令与所选卡：`hy-smi` 或 `rocm-smi`、观测到的 HCU 利用率和显存占用，以及 benchmark/profile 使用的 `HIP_VISIBLE_DEVICES=<idle-card>`。
- `PYTORCH_ROCM_ARCH`、DTK/ROCm、PyTorch 和 HIP 版本、设备名称及 `gcnArchName`。
- 准确的构建/安装、导入冒烟、正确性、benchmark 和 profiler 命令。
- 通过/失败阈值：正确性数值容差，以及相对基线的性能目标。
- benchmark 稳定性规则：预热与重复次数、汇总统计量（p50/p90/mean）、可接受噪声带和构成有效优化的最小变化量。

除非用户明确要求分开执行，且两个路径指向同一已编译扩展，否则不混用宿主构建和容器测试。

## LightOp 集成规则

新增或修改算子前，先检查最接近的现有算子族，并遵循其风格。

对于新的融合算子，先从名称、用户要求的语义或实现草图恢复组成部分。在设计新 kernel 前，先搜索 LightOp 中融合前的单算子及相关融合实现。以这些本地实现作为 API 形式、张量校验、分发/配置风格、正确性参考、benchmark 对照和性能预期的首选基线。如果 LightOp 没有对应的本地基线，记录搜索词和缺失情况，再采用 PyTorch 或独立参考实现。

主要源码位置：

```text
csrc/kernels/                 # 通用算子及按家族分组的子目录
csrc/mqa_logits/              # 按 dtype、paged、sparse 等分组
csrc/moe_gemm/
csrc/moe_marlin_gemm/
csrc/asm/
csrc/common/
csrc/include/
csrc/pybind/export.cpp
python/lightop/<category>/<wrapper>.py
python/lightop/__init__.py
setup.py
test/ci_tests/test_<op>.py
test/benchmarks/             # 同时复用算子族已有 benchmark
python/lightop/config.py
python/lightop/config_deepgemm.py
python/lightop/config_smooth_gemm.py
python/lightop/moe/config.py
python/lightop/tuning/
configs/
```

LightOp 的 HIP/C++ kernel 源文件使用 `.cu`。不要为 LightOp 算子创建或修改手写 `.hip` 源文件；本地构建产生的 `.hip` 视为生成产物，不作为需要纳入仓库的源码格式。

LightOp 开发规范门禁：

- 编辑前检查最接近的同类 LightOp 实现，遵循其目录、文件命名、C++ namespace、include 顺序、launch helper、宏/分发模式、错误检查习惯和格式。
- 遵循现有 Python wrapper 的导入、tensor/device/dtype 检查、连续性处理、可选参数、返回形状和回退/报错风格。已有本地模式时，不另创面向用户的 wrapper 模式。
- pybind 注册集中在 `csrc/pybind/export.cpp`，放在同类算子附近，遵循符号名、参数顺序、默认值和 `m.def(...)` 风格。API 头文件声明普通算子接口；kernel `.cu` 实现计算和 launcher。不要在这些头文件或 kernel 文件中放置 pybind 注册 helper 或 `pybind11::module_` 接口。
- 复用相关测试，遵循 CI 对种子、容差、dtype/shape、同步、skip 和命令行行为的约定。新增正确性测试文件放在 `test/ci_tests/`，并核对 CI 实际收集。长时间 benchmark 必须显式启用，不能在 import 时执行。
- 添加新表、分发分支、shape key、环境变量或 benchmark CLI 参数前，遵循已有 benchmark/config 约定。
- 不添加无关依赖、内嵌外部 kernel、生成源码、大范围重构、纯格式改动或其他项目的目录结构，除非用户明确要求且计划中说明原因。
- 优先扩展现有实现、模板、分发、wrapper 和 `.cu`，而不是复制近似 kernel 或增加平行 API。只有没有合适参考实现或现有文件无法承载要求时，才有理由新增 `.cu`；说明决策并遵循最接近的算子族规范。
- 新参数必须可选，默认值在 Python 和 pybind 两侧都保持旧调用、位置参数顺序、返回值和行为。专用计算方式的变化应加默认关闭的可选参数；省略或关闭时保留原路径。对支持输入保持原语义、且经过验证的通用优化，可以不增加开关而替换内部实现；不能悄悄将 workload 专用模式设为旧 API 的默认值。
- 不向 LightOp 源码树添加冗余源码/测试文件、benchmark 结果文件、任务报告或专门的 Markdown 文档。按需复用已有测试和维护中的 API 文档。实验输出和辅助脚本放在用户指定证据目录或忽略跟踪的循环状态中，不进入交付源码 diff。
- 如果确实需要优化说明，将简洁的理由、适用范围、取舍和复现命令写入对应测试文件的模块 docstring 或注释。不要单独创建优化 Markdown 文件；不要嵌入大量日志/结果，也不要仅为了容纳说明而新增冗余测试文件。
- 最终汇报前执行下面的四项评审。列出修改文件、相邻风格参考及有理由的偏离；确认没有生成 `.hip`、冗余实现、无关改动或实验产物进入源码 diff。

新增算子的常规清单：

- 对融合算子，检查每个组成单算子的 wrapper、binding、kernel、测试、benchmark 和配置，以及数据搬运或 epilogue 结构相似的相邻融合 kernel。
- 组成单算子存在时，以未融合的 LightOp 组合建立首次正确性和性能基线；否则采用最接近的 LightOp 实现加 PyTorch 参考。
- 先在 `csrc/kernels/` 或现有专用家族（如 `csrc/mqa_logits/`、`csrc/moe_gemm/`）中定位相近实现，再修改或扩展；只有现有文件无法承载时才新增文件。
- 在 `csrc/pybind/export.cpp` 中通过 `m.def(...)` 导出 C++ 符号。
- 复用最接近的 Python wrapper 并保留其张量校验风格。按实际包层级解析扩展导入，不把顶层相对 import 原样复制到分类子包。
- 新的面向用户 API 通过现有分类包中的 `lightop.<category>.<api>` 提供。保留历史顶层导入；不能用新增顶层导出来代替分类 API。
- 若新增源码目录，检查并更新 `setup.py` 中的 `_CSRC_DIRS` 显式清单及相关 include。保留非递归、按家族收集的方式；被 `#include` 引入的实现文件不能再作为独立编译单元，不改为整个 `csrc/` 的递归 glob。
- 性能依赖 shape/gfx 选择时，更新或补充 `python/lightop/config.py`、`python/lightop/moe/config.py` 或 `python/lightop/tuning/` 下对应配置/分发表。
- 现有 CI 正确性覆盖足够承载时进行扩展；否则新增聚焦的 `test/ci_tests/test_<operator_name>.py`。核对收集、断言及测试确实命中新路径。复用 benchmark，或在测试中提供显式性能选项；不要创建重复任务测试。辅助实验和输出不进入交付源码 diff；只有循环记录不能代替可执行 CI 覆盖。

仅优化现有算子时的清单：

- 找出当前 Python wrapper、导出的 C++ binding、kernel launch 路径、配置/分发分支及现有测试/benchmark。
- 保留公开 API，除非用户明确要求破坏性变更。
- 不因现有算子优化任务增加新的公开 API。
- 不创建无关算子文件或新算子族。
- 不修改无关算子族。
- 改动优先限定在目标 kernel、launcher、必要的分发/配置、针对性测试和 benchmark 覆盖。
- 在选定环境中做构建/导入验证、正确性、benchmark、profiling 和 tuning；安装仅限已授权范围。除非现有测试/benchmark 无法覆盖目标，否则不绕行完整的新算子集成流程。
- 改动范围限于目标任务所需的算子族、绑定、测试、benchmark 和 tuning 表。
- 复用用户指定或现有目标测试，扩展受影响路径的覆盖。确需新测试文件时放 `test/ci_tests/`，并验证 CI 收集。不要强制移动无关旧测试，也不要仅因任务有新名字就创建新文件。
- 优化退化或仅帮助非目标 shape 时，记录被拒绝的优化路线。

## 交付前的四项评审

这些检查针对用户要求的 diff，不针对无关历史代码。只请求 review 时，给出发现、迁移和参考实现建议；只有用户要求修复时才修改项目文件。读取适用 `AGENTS.md`，记录 Git 根目录、base/head、暂存/未暂存改动及相关未跟踪文件。使用 `git diff --name-status --find-renames`，沿“公开 API → wrapper/分发 → native 实现 → 构建 → 测试”核对。审查提交时读取 Git 对象，不混入脏工作区。使用 PR 目标 merge-base，或说明第一父提交/根提交的处理方式。

### 1. 测试位置与 CI 执行

新增测试文件放在 `test/ci_tests/`。复用覆盖，不重复建设；没有新文件不等于没有新行为需要测试。核对真实 CI runner 的收集和执行，包括 import、fixture、相对路径和用例清单。仅有 `__main__`、收集 0 个测试、无条件跳过、只打印错误没有断言，或用退出码掩盖失败，都不是正确性证据。对位置错误的新测试，给出原路径 → 目标路径及所需 runner/import 修改，移动前检查重名。耗时 benchmark 通过命令、选项或 marker 显式启用；默认 pytest 不运行长 benchmark、不加载模型权重。

### 2. 测试依赖与独立参考

检查直接和传递导入，包括局部 import、`importorskip`、动态 import、`conftest.py`、helper 和 fixture。对 SGLang、vLLM、AITer、LMSlim 等大依赖，明确用到的函数/常量，以及为何 CI 需要整个包。数学、数据生成、量化/packing 和计时的语义可独立复现时，优先采用 PyTorch/标准库参考。明确 shape、dtype、scale 方向、舍入、饱和、mask 和布局；不猜测缺失语义，也不将待测算子/核心实现作为自己的 oracle。替换依赖时，提供最小兼容可调用实现；原实现可用时对比代表输入。将必要的框架协议/布局集成测试与核心数值测试分离；不要声称参考实现代替了集成覆盖。不删断言、不放宽容差、不 skip 来掩盖依赖失败。区分新阻塞依赖与原有依赖的可选精简。

### 3. 正确性、性能与构建身份

将每个新公开 API 或行为映射到实现、正确性测试和 benchmark；别名、wrapper、文件、kernel 变体分别计数。使用独立参考及有效断言，覆盖受影响的 dtype/shape、tile 边界、尾块、支持的 stride、空输入、索引、输出别名和原地副作用。按真实契约检查量化值和 scale、有限值、NaN/Inf、饱和/舍入和布局。容差遵循契约，不统一套阈值或只看余弦相似度；近零参考考虑绝对误差，避免 NaN 悄悄通过比较。新一轮输入要考虑已有原地修改。

对照旧实现或有意义的已有组合，测量等价工作，排除首次编译/加载。明确 kernel-only、包含分配/转换的调用或端到端计时；同步真实工作所在 stream，报告重复计时和离散程度。把 profiler 开销与 benchmark 延迟分开。性能退化率为 `(new - base) / base * 100%`，加速比为 `base / new`；不要虚构性能阈值。验证规模与受影响行为匹配：绑定重构不自动要求完整 kernel 优化循环。

通过源码/构建配置、产物 SHA、`lightop.__file__`、`lightop.op.__file__` 及 Python/PyTorch/HIP/gfx 身份确认待测构建。“安装成功”或绿色 CI 本身不证明测了哪个源码。相关时使用兼容基线工具的 check 模式，不重写基线来隐藏非预期 API 变化。分别记录测试是否存在、CI 收集、实际执行、加载构建及 pass/fail/skip/error/timeout。没有目标硬件时，只能给静态发现和针对性命令，不能宣布数值/性能通过。

### 4. 算子分类与公开 API

新公开入口采用 `lightop.<category>.<api>`，必须由包实际导出，不能只有 C++ 绑定或目录名称。

| 类别 | 主要职责 |
| --- | --- |
| `gemm_ops` | GEMM、Grouped GEMM、量化矩阵乘 |
| `moe` | 专家路由、dispatch/combine、专家计算及重排 |
| `attention` | Attention、MQA logits、RoPE、注意力索引/选择 |
| `kvcache` | KV 分配、写入、读取、分页和存储布局 |
| `norm` | RMSNorm、LayerNorm、GroupNorm 及归一化融合 |
| `activation` | 激活函数及 SiLU/SwiGLU 类融合 |
| `quant` | 独立量化、反量化、scale 和格式转换 |
| `sampling` | 采样 Top-K/Top-P、概率过滤和 token 选择 |
| `vision` | 图像采样、插值和上采样 |
| `loss` | CTC、RNNT 及其他损失/梯度 |

按主要语义分类：Attention Top-K 不自动归为 sampling；量化 GEMM 通常仍归 `gemm_ops`。保留历史顶层 API、导入路径、签名、默认值、返回顺序、`__all__` 和延迟加载。不仅为匹配包名而移动 kernel。现有 `tensor`、`nn`、`ops`、`utils` 或 `tuning` 目录不自动批准一个新公开类别。如果十类都不合适，在提出新类别前说明职责、相关类别不适用的原因、预期关联 API、边界、导出和测试组织。

### 多贡献者集成检查

- 历史别名与分类后的别名共用一套实现；优先重新导出同一个 wrapper，而不是复制其逻辑。可复用 helper 放在所属算子族附近，内部 tuning 细节不要暴露成公共 API。
- 在现有签名/docstring 中明确可选参数的语义：允许的值、默认值/省略时的行为、输出所有权和回退方式。验证旧调用、显式默认值、启用路径，以及不满足快速路径条件的输入。新增参数不能重新解释原有位置参数。需要时复用现有配置/环境变量处理方式；不能仅为了绕过逐次调用的 API 决策就新增全局开关或环境变量。
- 对 dtype、shape、stride、对齐、device/gfx 和模式设置精确的特化条件。不属于快速路径适用域的输入保留原有路径；本来就超出 API 契约的输入可以明确拒绝。不得广泛捕获运行时/kernel 错误后悄悄换一种计算方式重试。
- 保持支持范围内各 package/native 版本的导入行为。可选 native 符号可能缺失时，沿用已有的能力/版本判断方式；不能因为新可选 helper 不可用就使无关分类的 import 失败。必需功能缺失时不得假装成功。涉及导入状态或注册时，在全新进程中做兼容性检查。
- 性能/显存结论要计入临时转换、`.contiguous()`、复制、`.item()`/主机同步、分配和 workspace 生命周期。不要新增隐藏的持久缓存或无上限的暂存空间；workspace 增长发生变化时应记录。allocated/reserved 峰值以及 graph/private pool 对结果有实质影响时，要分别列出。这些是受影响路径检查，并不要求每个算子都跑模型 benchmark。
- 模板实例化数量限定在实际支持的特化矩阵内，复用公共代码；新增大量变体时考虑编译时间和二进制大小。不得仅为缩短编译或改善一个 shape 就删除已支持的 dtype/架构，或全局修改数学编译选项。核验相关的当前 CI 版本/架构分支及包内资源。
- 阅读当前 CI workflow/runner，不能只看历史迁移文档。如果 CI 构建的是合并后的代码树，测试的却是另一 checkout，应记录两者 SHA 并确认实际使用的产物。源码 diff 聚焦算子、绑定/导出、必要的构建/配置及可执行测试；证据总结放在 PR/任务回复中，不另生成报告文件。

### 受影响路径检查与审查输出

只检查本次 diff 涉及的风险；在认定 kernel 缺少检查属于缺陷前，先考虑 wrapper/caller 已经提供的保证：

- Native 对 dtype/shape/stride/对齐/device/容量的假设；读写 mask、索引窄化/溢出、负数哨兵、page 与 Top-K 重映射。
- Workspace 初始化、输出是否完整写入、异步生命周期、缓存 identity、并发写入、barrier，以及目标 wave/活跃 lane 规则。
- Kernel、复制、BLAS handle 和分配使用的当前 device/stream；跨 stream 顺序和生命周期。只有承诺支持 graph 的路径才检查 graph 更新/capture 与首次使用初始化；只有通信受影响时才做多 rank 检查。名称含 CUDA 的 PyTorch guard 在 HIP 下也可能完全有效。
- 浮点格式、累加、scale 方向、舍入、融合/cast 顺序、softmax 稳定性、epsilon、routing 和索引语义。只有训练接口才检查 backward、保存的 tensor/RNG；不要对所有低精度 kernel 强制要求 double gradcheck。
- 源文件 glob、include、符号、签名、版本/gfx 分支、wheel 资源与实际构建入口；不要机械修改所有旧 setup 脚本。检查 lazy import、同名 module/package 遮蔽以及 custom-op 注册冲突；相关时在全新进程中检查导入顺序。仅从源码树运行不能证明 wheel 完整。

四项检查均用“符合 / 需整改 / 不适用 / 待验证”报告，除非用户要求其他格式，否则使用中文。具体问题应给出严重性、文件/行号、触发条件、影响、证据、最小修复及验证方法；区分本次 diff 引入的退化、旧问题和测试策略缺口。P0 指大范围严重不可用/数据损坏；P1 指受支持路径结果错误、内存损坏、死锁、关键兼容性破坏或已证实的严重性能退化；P2 指局部缺陷、明确的测试/分类缺口或可复现的有限退化；P3 指不阻塞合入的可维护性建议。合并相同根因的问题。仅在有用时给出迁移/依赖/覆盖映射，说明已执行和未执行的检查；没有确认缺陷时明确说明。不要扩展为无关重构或许可证/品牌审查；本次 diff 修改第三方代码或打包时才做相应源码/许可证检查。结果放在任务回复/指定证据位置，不在 LightOp 中新增 Markdown 报告文件。

## HCU/ROCm 默认规则

- LightOp 测试中的 PyTorch `torch.cuda` namespace 视为 ROCm 运行时接口。
- 优先 `hipcc`/ROCm 扩展构建，不采用仅适用于其他平台的编译路径。
- 提交的 LightOp kernel 源码扩展名使用 `.cu`，不手写提交 `.hip` 算子文件。构建生成的 `.hip` 保留为构建产物，不作为源码修改。
- 遵守 `PYTORCH_ROCM_ARCH`；未设置时从 `torch.cuda.get_device_properties(0).gcnArchName` 推断 gfx。
- 本地 LightOp setup 已采用默认 `gfx928;gfx936;gfx938` 时优先沿用，除非用户给出更窄目标。
- 不在 LightOp 路径中引入 CUDA-only 头文件、PTX/SASS 假设、CUTLASS/CuTe、Nsight Compute、TMA/WGMMA/tcgen05 或其他平台的架构规则。
- 借鉴 CUDA 来源时，转换为 ROCm 概念，并记录来源仅用于跨平台启发。

可用的环境探测命令：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("device:", torch.cuda.get_device_name(0))
print("gcn:", torch.cuda.get_device_properties(0).gcnArchName)
PY
hipcc --version
```

## 性能设备门槛

任何 benchmark 或 profiling 命令执行前，在目标环境检查卡状态，将运行固定到空闲卡。

- 性能命令前立即执行 `hy-smi` 或 `rocm-smi`。
- 选择 HCU 利用率和显存占用都低的卡。没有足够空闲、能产生稳定结果的卡时，延后运行，或记录结果受噪声影响，不能作为最终证据。
- benchmark 和 profiler 命令加 `HIP_VISIBLE_DEVICES=<card>` 前缀。基线/候选配对比较使用同一张卡，除非比较目的就是跨设备行为。
- 在尝试台账或 profile 产物目录记录所选卡、HCU 利用率、显存占用和准确命令。
- 首次优化编辑前，在同一执行环境和所选卡上校准设备带宽。在 `.lightop-kernel-agent/device-bandwidth.txt` 记录实际读、写、复制/读写和简单 triad 带宽、缓冲区大小、dtype、所选卡和命令。将其作为判断用户有效带宽目标是否合理的基线。

示例：

```bash
hy-smi || rocm-smi
bash <skill-root>/scripts/measure-device-bandwidth.sh \
  --docker <container> \
  --workdir <container-lightop> \
  --hip-visible-devices <idle-card> \
  --output .lightop-kernel-agent/device-bandwidth.txt
HIP_VISIBLE_DEVICES=<idle-card> python test/<family>/benchmark_<op>.py
HIP_VISIBLE_DEVICES=<idle-card> hipprof python test/<family>/benchmark_<op>.py
```

直接在宿主执行时，省略 `--docker`，传入 `--workdir <lightop-root>`，或从 LightOp 根目录运行脚本。

设备选择门槛和设备带宽校准没有记录时，不将性能数字作为可采取行动的结论；用户明确接受证据缺失时除外。

## 工作流

### 阶段 1：检查与计划

1. 定位 LightOp 根目录和目标算子族。
2. 恢复 `K`、`R`、`W`、目标 gfx 架构、基线命令和成功阈值。
3. 检查已有 wrapper、绑定、kernel、配置表、测试及 benchmark。
4. 新融合算子按所需名称、名称词项、组成单算子名和语义搜索融合前的单算子及相关融合 kernel。记录选择的基线：未融合 LightOp 组合、相近融合实现、PyTorch 参考，或明确记录“未找到本地基线”。
5. 首次优化编辑前，建议查询 `lightop-kernel-knowledge`，了解本地 LightOp 模式、ROCm/HCU 上游证据、Hygon/HCU 源码参考和内置语料中的可迁移思路。能影响首次实现路线时就使用。
6. 在首个正式实现谱系开始前，在闭环状态中写简洁的研究摘要。
7. 编辑代码前定义 benchmark 契约：准确目标 shape、dtype/layout/连续性、axis/mode/epsilon、有效带宽公式、预热与重复次数、汇总统计量、噪声带，以及基线和候选将使用的 benchmark 命令。

### 阶段 2：实现与验证

1. 首次优化编辑前建立基线矩阵：运行正确性，校准所选卡的设备带宽，再在所选空闲卡上对目标工作负载做足够重复，报告 p50/p90 或 mean、有效带宽、方差/噪声、卡状态、设备读/写/复制带宽及命令行。保存到已有尝试台账；确需简短优化说明时放入相关测试文件。
2. 做能满足当前任务与验收要求的最小 LightOp 源码修改。
3. 按目标架构构建 LightOp。
4. 运行目标正确性测试。在对 `W` 做 benchmark 前执行性能设备门槛，并用 `HIP_VISIBLE_DEVICES` 固定设备。
5. 记录每个候选结果：正确性失败、构建失败、回退、平台期或改善。
6. 每个通过正确性的优化候选先运行普通同步 benchmark；下次修改 kernel 或 dispatch 前，对同一代表性目标 shape 收集 `hipprof --pmc` 证据。采集缓存计数器、LDS/bank-conflict 线索、occupancy/资源信号，以及准确命令输出或工具不支持的原因。用 `hcu-profiler-report` 将这些证据转为下一次编辑。
7. 每候选 PMC 证据和 benchmark 仍无法解释下一次编辑时，调用更深入的 `hcu-profiler-report` 分析。
8. 首个通过正确性的候选未达到性能阈值时，继续进入性能分析与调优，不能宣布完成。

### 阶段 3：调优与集成

1. 为 `W` 建立性能映射。
2. 每个优化谱系聚焦一个主要假设，如向量化访存、block/thread 映射、寄存器压力、LDS 布局、occupancy、launch 配置、epilogue 融合或 shape 特化。不要在一个候选中混合多项无关技术；代码结构导致无法拆分时，记录原因。
3. 只有实测表明不同区间需要不同选择时，才添加 shape/gfx 感知的 dispatch 或配置表项。
4. 对所有触及的 dtype/layout/mode 重跑正确性。
5. 重跑定义 `W` 的 benchmark 及相近回归保护用例。
6. 每次通过正确性的优化编辑后重采每候选 `hipprof --pmc`。下次编辑前总结缓存行为、LDS/bank conflict、occupancy/资源压力，以及一个由 profiler 支持的下一步动作。
7. 候选谱系正确性失败、收益未超过噪声/稳定性阈值或仅改善非目标 shape 时，从最终路径拒绝或回退该谱系。缺少 profile/resource/ISA 时区分任务验收缺口与方法归因缺口，按 HIP 优化规则记录降级或 inconclusive；不能宣称缺失证据已通过。保留拒绝及归因记录，不静默覆盖。
8. 达到目标后做最终保护性验证：目标正确性、所选卡上重复的目标 benchmark，以及适用的邻近 shape/dtype 回归。
9. 总结最终代码路径、fallback 行为、不支持区间和剩余风险。

## 性能目标规范

用户给出带宽、时延、吞吐或加速目标时，将阈值纳入验收契约。

- 最佳正确性通过候选仍未达标时，不宣称完成。只能报告为当前最佳结果，并给出瓶颈证据和下一步。
- 每个通过正确性的候选记录 shape、dtype、layout/mode、kernel 或 dispatch 配置、实测带宽/时延、比较基线，以及改善、回退或平台期的原因。
- 每个通过正确性的优化候选，在决定下次编辑前记录代表性工作负载的配对 `hipprof --pmc` 证据。摘要必须讨论缓存行为、内存/缓存流量、LDS 或 bank-conflict 证据、occupancy/资源压力，并且只给出一个下一步 LightOp kernel、launcher、dispatcher、config 或 benchmark 编辑。已安装 DTK 无法提供某个 PMC 计数器或 occupancy 信号时，记录准确命令和失败，不能猜测。
- 性能改善只有超过计划定义的 benchmark 噪声/稳定性阈值，才算有效。变化在噪声带内时记为 inconclusive 或平台期，不能算优化胜利。
- 首个通过正确性的候选未达标后，停止前必须开展性能分析与调优闭环。
- 在任务预算允许且有新证据时探索不同优化谱系；不能为凑次数重复失败方法。达到预算、连续三轮无实质改善或目标不可达时，按 HIP 优化规则诊断并停止。
- 连续两个通过正确性的候选未达标时，下一次 kernel 或 dispatch 编辑前必须同时完成：
  - 一次 `lightop-kernel-knowledge` 研究，覆盖目标算子族的本地 LightOp 模式（归约/归一化任务才查 layernorm/rmsnorm/fused-norm）、相关 ROCm/HCU 上游证据，以及语料中可迁移的归约/向量化思路。
  - 针对代表性目标 shape 的 `hcu-profiler-report` 摘要。
- 第二次通过正确性的优化尝试，相对相关父版本或基线提升不足 5% 时，下次优化编辑前必须进行深度 `hcu-profiler-report` 分析。内容包括 `hipprof` PMC all 模式、可用时的 SQTT JSON、`dccobjdump` 反汇编、code-object 资源用量，以及明确的 LDS/寄存器/occupancy 证据，或逐项记录不可用原因。
- 该门槛后的下一次编辑必须只指定一个具体的 LightOp kernel、绑定、dispatcher、config 或 benchmark 修改，并引用所依据的知识和 profiler 证据。
- 不凭直觉、源码检查或预期硬件行为宣称优化有效。进入优化台账必须有实测且通过正确性的 benchmark、基线或父版本比较，以及每候选 PMC/profile/资源证据。
- 必需调优谱系完成后仍未达标时，总结最佳候选、失败谱系、profiler 瓶颈类别、不支持区间和最可能的下一步工程投入。不能将算子描述为性能已完成。

## 必需的闭环状态

证据放在交付的 LightOp 源码 diff 之外，优先使用用户指定目录，否则使用被忽略的本地闭环状态。不在算子树新增独立结果、报告、README 或重复 Markdown；按需复用既有记录，不创建空文件。以下运行时路径按需使用：

```text
.lightop-kernel-agent/plan.md
.lightop-kernel-agent/research-digest.md
.lightop-kernel-agent/attempt-ledger.md
.lightop-kernel-agent/state.json
.lightop-kernel-agent/optimization-ledger.md
.lightop-kernel-agent/lineage.jsonl
.lightop-kernel-agent/performance-map.json
.lightop-kernel-agent/tuning-decisions.md
.lightop-kernel-agent/profile-artifacts/
```

创建状态前确认 `.lightop-kernel-agent/` 已被忽略。默认用户缓存位于仓库外；若将 `XDG_CACHE_HOME` 指向仓库内的 `.cache/`，则使用已在 `.gitignore` 忽略的 `.cache/lightop-kernel-agent/`。除非用户明确要求跟踪证据产物，否则不提交闭环状态。

每个已做 benchmark 的候选之后，更新已有尝试台账，不另生成优化 Markdown。必要的简短优化说明写入相关测试文件。来源、方法、ISA 和消融条目也由同一既有台账承载，不另建优化说明文件。以下是已有台账的记录模板，不是向 LightOp 新增报告的要求：

```markdown
# Kernel 优化日志

## 基线矩阵
- 目标工作负载：
- 有效带宽公式：
- 设备门槛：
- 设备带宽校准：读/写/复制/triad 带宽、缓冲大小、所选卡、命令：
- 构建/测试/benchmark 命令：
- 基线 p50/p90/mean、方差/噪声：

## 迭代 <N>：<一个主要假设>
- 假设：
- 修改文件：
- 代码/配置修改：
- 构建命令/结果：
- 正确性命令/结果：
- 设备状态与 HIP_VISIBLE_DEVICES：
- Benchmark 表：基线/父版本/候选、p50/p90/mean、有效 BW、变化量、噪声阈值：
- 每候选 PMC/profile/资源证据：缓存行为、LDS/bank conflict、occupancy/资源压力、不可用计数器：
- 决策：keep | reject | inconclusive
- 原因：
- 下一步：
```

## 构建、测试与 Benchmark

在所选环境的实际 LightOp 根目录构建，使用当前仓库支持的入口和版本分支。构建入口以当前仓库为准；生成发布 wheel 时使用 [build-hcu-wheel](../../build-hcu-wheel/SKILL.md) 所述的 `scripts/build_hcu_wheel.sh`，不在本流程重复维护编包步骤。仅当要求支持的矩阵涉及旧入口时才使用旧入口。在已经授权安装的隔离环境中，例如：

```bash
PYTORCH_ROCM_ARCH='gfx928;gfx936;gfx938' python setup.py install
```

所选执行环境为 Docker 时，使用用户提供的容器内路径包装同一命令：

```bash
docker exec <container> bash -lc 'cd <container-lightop> && PYTORCH_ROCM_ARCH="gfx928;gfx936;gfx938" python setup.py install'
```

各次尝试之间保留现有 `build/` 目录，让增量 extension 构建复用已有编译输出。`python setup.py install` 可能创建或更新 `build/bdist.*`、`build/lib.*`、`build/temp.*` 等正常 setuptools 子目录；保留它们供下次构建复用。在正常构建/测试/调优循环中，不得删除、改名、移动或重建 `build/` 及其子目录，除非用户明确要求 clean build，或已证明构建缓存过期/损坏。

没有已证实的缓存问题或用户要求时，不删除构建输出；不混用主机/容器构建身份；不将替换共享服务安装包作为验证的附带动作。使用包隔离 runner 前先检查其行为。

构建或获授权安装后，在同一验证环境中导入所选产物，同时记录 Python 包路径和 native 路径：

```bash
python - <<'PY'
import torch, lightop
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("lightop:", getattr(lightop, "__file__", "unknown"))
print("native:", getattr(lightop.op, "__file__", "unknown"))
print("device:", torch.cuda.get_device_name(0))
print("gcn:", torch.cuda.get_device_properties(0).gcnArchName)
PY
```

涉及 wheel 改动时，在源码 checkout 之外验证导入/资源查找，避免本地文件掩盖包内容缺失。纯绑定位置调整且计算代码未变时，针对性的构建/import 检查即可；kernel/数学/dispatch 修改则需要受影响的数值与性能检查。

先检查相关测试的 import 和副作用，再执行范围最小的相关测试：

```bash
python -m pytest --collect-only -q test/ci_tests/test_<operator_name>.py
python -m pytest -v -p no:cacheprovider --tb=short test/ci_tests/test_<operator_name>.py
```

复用已有 CI 测试；只有必要时才在该位置新增针对性文件。用户指定的已有旧测试仍可用于回归覆盖；新文件遵循 CI 目录规则。正确性测试需要独立参考实现和断言，并覆盖旧/默认路径及新增可选路径。不要为每个候选新建测试入口或结果文档。

耗时的性能验证单独、显式执行。复用已有 benchmark 契约；如果新增 benchmark 开关，可以从预热 10 次、计时 100 次起步，并根据工作负载和噪声调整。使用同步 GPU event 或同步 wall time，报告延迟统计量及已定义的有效带宽，保持测量边界可比。import 阶段和默认 pytest 都不能运行长 benchmark。对于实现了显式开关的测试，示例如下：

```bash
HIP_VISIBLE_DEVICES=<idle-card> python test/ci_tests/test_<operator_name>.py --benchmark
```

辅助实验及输出产物放在指定证据目录或被忽略的循环状态中。优先参考已有 LightOp 算子组合/PyTorch/字面值参考实现；框架特有参考实现需要前述依赖审查。只有循环日志不能替代可执行的正确性覆盖。

最终回复或指定证据记录必须用简洁 Markdown 表展示最终验证，至少包含：测试文件、参考实现、shape、dtype、正确性结果、平均耗时（us）、有效带宽、所选卡和命令。

每次 benchmark 前都执行本 skill 的性能设备门禁：记录 `hy-smi` 或 `rocm-smi`，选择低利用率/低显存占用卡，并用 `HIP_VISIBLE_DEVICES=<idle-card>` 执行 benchmark。基线和候选测量复用同一所选卡。

每个通过正确性的优化候选，在下一次修改前对相同的代表性 benchmark shape 采集 PMC 证据。使用同一所选卡，产物存放在 `.lightop-kernel-agent/profile-artifacts/<version>/` 下；按工具支持情况执行如下命令变体：

```bash
HIP_VISIBLE_DEVICES=<idle-card> hipprof --pmc --pmc-type 3 \
  -o .lightop-kernel-agent/profile-artifacts/<version>/hipprof-pmc-all/pmc \
  python test/<family>/benchmark_<op>.py
HIP_VISIBLE_DEVICES=<idle-card> hipprof --pmc-read --pmc-type 3 \
  -o .lightop-kernel-agent/profile-artifacts/<version>/hipprof-pmc-all/pmc-read \
  python test/<family>/benchmark_<op>.py
HIP_VISIBLE_DEVICES=<idle-card> hipprof --pmc-write --pmc-type 3 \
  -o .lightop-kernel-agent/profile-artifacts/<version>/hipprof-pmc-all/pmc-write \
  python test/<family>/benchmark_<op>.py
```

利用 PMC 摘要检查缓存行为、内存/缓存流量、LDS 或 bank 冲突、occupancy/资源压力，并确定下一个具体修改。已安装 DTK 不支持某个计数器、`--pmc` 模式或 occupancy 信号时，将尝试的命令及错误输出记录在产物目录。

不能仅凭构建成功宣称数值或性能通过。Kernel 优化应核验所选构建/import 身份、受影响路径的正确性、benchmark 对比，以及要求的每候选 PMC/profile 证据。纯绑定或组织结构重构则需要相关构建/import/API 和受影响路径检查；不要虚构性能收益，也不要强行开展无关 GPU tuning。异步 HCU 工作未同步时，不能用 Python wall-clock 计时声称加速。

基线与候选比较使用相同的 benchmark 命令、所选卡、工作负载 shape 和有效带宽公式。工作负载契约改变时，重新建立基线矩阵并解释原因。

## 性能分析

每个通过正确性的优化候选后，自主调用 `hcu-profiler-report`，解释强制要求的 `hipprof --pmc` 采集，并生成下一次编辑。摘要可简洁，但必须解释缓存行为、LDS/bank-conflict 线索、occupancy/资源压力，以及且仅有一个 profiler 支持的下一步动作。

出现以下任一情况，从每候选必需 PMC 分析升级为深度分析：

- 基线 benchmark 已通过，但没有 profile 摘要。
- 正确候选相对基线或先前最佳在 +/-2% 范围内。
- 正确候选在重要 shape 上回退。
- benchmark 进入平台期，下一次编辑不明确。
- 首个通过正确性的候选未达到用户性能目标。
- 连续两个通过正确性的候选未达标；下一次 kernel 或 dispatch 编辑前，将本次分析与 `lightop-kernel-knowledge` 研究配对完成。
- 第二次通过正确性的优化尝试相对父版本或基线提升不足 5%。这是强制深度分析门槛，不是启发式建议。
- 候选明显快于预期，需要解释。
- 复核者要求性能分析证据。

profile 产物保存在 `.lightop-kernel-agent/profile-artifacts/` 或用户指定的证据目录。每份摘要末尾必须且只能给出一个具体下一步 kernel 修改，或明确说明为什么分析结果无法支持行动。

触发第二次优化 <5% 门槛时，摘要必须包含 `hipprof` PMC all、支持时的 SQTT JSON、`dccobjdump` 反汇编、code-object 资源用量，以及 LDS/寄存器/occupancy 证据。缺少工具时，记录准确的尝试命令，不能用猜测替代缺失证据。

## 计划要求

用普通 Markdown 写入 `.lightop-kernel-agent/plan.md`，为以下内容设置验收项：

- LightOp 根目录、目标算子族、公共 API 和修改文件。
- 新融合算子的融合前单算子搜索结果、相关融合实现搜索结果，以及选定基线/参考路径。
- 明确的 `K`、`R`、`W`、目标 gfx 架构、基线命令、比较目标及严格排除范围。
- 工作负载契约：目标 shape、dtype、layout/连续性、axis/mode、epsilon 或其他数学参数、有效带宽公式和准确成功指标。
- 明确的 `E`：Docker 容器/镜像或直接宿主模式、环境内 LightOp 路径、可见设备选择、安装/冒烟/正确性/benchmark/profiler 命令和通过阈值。
- 设备门槛：`hy-smi` 或 `rocm-smi`、空闲卡选择标准、benchmark/profile 必需的 `HIP_VISIBLE_DEVICES=<idle-card>` 前缀，以及状态输出保存位置。
- 首次优化编辑前的设备带宽校准：准确命令、所选卡、缓冲区大小、dtype、实测读/写/复制/triad 带宽、产物路径，以及如何约束或解释目标有效带宽阈值。
- 对 `W`、边界情况、dtype/layout/mode 边界和基线/参考一致性的正确性覆盖。
- LightOp 开发一致性清单：采用哪些同族文件作风格参考、遵循的 wrapper/绑定/config/测试/benchmark 约定、允许偏离及禁止修改的文件。
- 构建命令、ROCm/DTK/PyTorch 版本、`PYTORCH_ROCM_ARCH` 和设备元数据。
- benchmark 方法：预热、重复、同步、逐 shape 计时、适用的 p50/p90 或 mean、方差/噪声带、最小有效变化量和环境元数据。
- 测试策略：复用受影响的已有覆盖；新正确性文件放在 `test/ci_tests/`，确认 CI 收集且有有意义的断言。覆盖旧/默认行为及显式启用行为。性能测试显式启用，采用独立可比基线与同步计时，不在 import/默认 pytest 时运行。结果/helper 留在交付 diff 之外，不新增冗余测试或 Markdown 报告。
- 四项审查：CI 位置与执行、大型依赖最小化、独立正确性/性能覆盖及实际构建身份、十类导出和向后兼容；仅检查受影响的内存、数值、device/stream、graph、autograd 和构建资源风险。
- 每个通过正确性的优化编辑之后采集 `hipprof --pmc`，记录产物路径、卡、代表性 shape、缓存计数器或缺失原因、LDS/bank-conflict 证据、occupancy/资源解释和 profiler 支持的下一次编辑。
- 首次优化编辑前必需的基线矩阵：卡状态、设备带宽校准、重复计时、有效带宽、噪声/稳定性估计。
- 迭代规范：每个谱系只围绕一个主要假设，明确 keep/reject/inconclusive 决策规则、任务时间或迭代预算、平台期诊断和停止条件；未给预算时根据任务范围记录合理预算，不能无限循环。
- 研究摘要：本地 LightOp 模式，以及实质影响路线的上游/源码证据。
- 每个候选的尝试台账。
- 每个 benchmark 候选之后更新已有尝试台账；必要的简短优化说明放在相关测试文件，不新增独立优化 Markdown。
- 优化台账只接纳正确且实测改善的候选。
- 性能目标规范：首次未达标触发调优；连续两次正确性通过但未达标时，下次编辑前同时补 `lightop-kernel-knowledge` 研究和 `hcu-profiler-report` 证据；未达到目标不能报告完成。
- 低收益规范：第二次通过正确性的优化提升不足 5% 时，下次编辑必须等待深度证据：`hipprof` PMC all、可用时的 SQTT JSON、`dccobjdump`、code-object 资源用量及 LDS/寄存器/occupancy 分析。
- `W` 包含多个区间时的调优决策与 dispatcher/config 更新。
- 最终正确性矩阵、benchmark 矩阵、fallback 路径、不支持区间、经相关 CI/benchmark 入口完成的最终达标保护性验证、简洁结果表、LightOp 一致性复核和剩余风险。

## 直接执行与恢复

完成计划和基线后，直接按阶段 1–3 执行，不调用外部循环启动器。每轮顺序为：profile 当前最佳 → 选择主要假设 → 实现候选 → 构建/正确性 → 同卡配对 benchmark → profile/ISA/必要消融 → keep/reject/inconclusive → 更新状态。

恢复任务时读取 `plan.md`、`state.json`、研究摘要及最后一轮台账，核对当前源码哈希、已安装扩展路径和设备/工具链。环境或工作负载发生变化时重新测基线，不能继承旧环境的性能结论。没有成功编译或通过正确性的候选不能晋升。

停止条件与方法归因见 [HIP 优化规则](hip-optimization-rules.md)。仅当用户请求版本管理操作时，暂存或提交相应修改。
