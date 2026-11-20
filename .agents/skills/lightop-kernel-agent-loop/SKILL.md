---
name: lightop-kernel-agent-loop
description: 在 HCU/ROCm 上新增或优化 LightOp 算子：恢复语义与工作负载契约，依据 HIP 性能证据选择优化方法，完成实现、构建、正确性、基准测量、ISA 核验和消融归因。
---

# LightOp 算子智能体闭环

直接执行证据驱动的 HIP 优化闭环，不使用 Humanize、RLCR、Stop hook 或外部复核 CLI 驱动算子优化。

开始前阅读[完整工作流](references/full-workflow.md)和[HIP 优化规则](references/hip-optimization-rules.md)。前者定义 LightOp 集成与验收，后者将 `kernel-hip-handbook` 和 `kernel-hip-optimize` 的知识转化为本闭环的选法、验证与停止规则。命令路径相对于对应 Skill 根目录解析。本副本随 LightOp 源码维护，属于开发工具，不参与算子编译或 `import lightop`。

## LightOp 集成约束

优先复用现有实现、模板、wrapper 和 `.cu` 文件。新增参数必须可选，保留旧调用与默认行为；专用计算方式通过默认关闭的可选参数控制，经过验证且保持原语义的通用优化除外。pybind 注册集中到 `csrc/pybind/export.cpp`，新公共 API 经已有分类包导出，新正确性测试放在 `test/ci_tests/`。

不向 LightOp 树添加冗余 kernel、测试、结果文件或任务专用 Markdown。必要的优化说明放在相关测试文件的模块 docstring 或注释中，不单独新增 Markdown 文件。交付前执行完整工作流中的[集成与四项审查门禁](references/full-workflow.md#lightop-集成规则)，包括依赖、兼容性和实际构建身份检查。

## 必须执行的优化规则

1. 恢复 K/R/W/E，核实实际 gfx、工具链、设备状态和基线；先正确性、再同卡配对计时和 profiling。
2. 根据 FLOPs/读写字节、实测带宽、PMC 与资源信息区分 compute/memory/latency/launch 瓶颈。按证据选择结构重构、访存、wave64 归约、LDS、寄存器、流水线或 shape 特化；不机械套用手册参数或收益数字。
3. 每个候选围绕一个主要假设，记录来源、适用条件、预期指标变化和父版本。保留当前最佳可回退实现，候选按顺序在同一空闲卡验证。
4. 保持 LightOp `.cu` 源码、wrapper/绑定/config 及当前仓库支持的构建入口与版本分支；需要生成发布 wheel 时复用 [build-hcu-wheel](../build-hcu-wheel/SKILL.md)，安装限定在授权范围内。矩阵路径优先考虑本工程已支持的 CK Tile/HCU 实现；不将独立 optimizer 的 `solve(...)` scaffold 直接作为 LightOp 集成结果。
5. 每个候选经构建、正确性、同步 benchmark、profile 后决策。涉及 builtin、向量指令、MMAC 或流水线的声明，还需目标编译探针及最终二进制的 `dccobjdump` 证据。
6. 分开判定“候选实测加速”和“方法贡献成立”。对组合优化做受控消融；无关符号、空 dump 或占位消融不能证明有效性。
7. 更新 `.lightop-kernel-agent/` 状态；达到目标并完成回归才报告达标。预算耗尽、证据不足或平台期按详细规则停止并报告当前最佳及未达标项。

## 配套知识与性能分析

研究使用内置 [lightop-kernel-knowledge](assets/lightop-kernel-knowledge/instructions.md)，性能分析使用 [hcu-profiler-report](assets/hcu-profiler-report/instructions.md)。两者包含查询、分析必需的脚本和离线来源索引；命令相对于各自目录运行。知识查询需要 Python 3.10+、PyYAML 6+；设备采样工具使用目标环境的 DTK。索引直接读取随仓库提供的 `index.jsonl`，不解压、不写入用户缓存；证据缓存默认位于 `~/.cache/lightop-kernel-agent/`，也遵循 `XDG_CACHE_HOME`。查询默认离线，只有显式获取尚未缓存的证据才访问网络，详见[语料与缓存说明](assets/lightop-kernel-knowledge/references/corpus-packaging.md)。抓取、重建索引和批量仓库维护工具保留在 hcu-infer-pilot。

深度优化按 [HIP 优化规则](references/hip-optimization-rules.md) 中的来源映射读取同级 `kernel-hip-handbook` / `kernel-hip-optimize`。它们是扩展知识来源，不是启动依赖；缺失时仍可执行本文及内置规则，但不能声称读取了缺失资料或调用不存在的脚本。

流程说明、示例说明和报告标题统一使用 HCU 或“目标平台”表述。软件项目与接口标识按实际调用保留；原始版权、许可证和来源数据不得改写归属。
