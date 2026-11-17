# LightOp HIP 优化规则

本规则吸收 `kernel-hip-handbook` 的瓶颈分析、资源调优、结构重构与 shape 特化，以及 `kernel-hip-optimize` 的方法选择、候选验证、ISA 和消融归因。LightOp 集成和测试仍遵守 [完整工作流](full-workflow.md)。

## 1. 来源与适用边界

下列为可选的扩展知识来源，不随此精简包分发。仅在已安装或取得相应来源时按实际路径读取；缺失时使用本文件的内置规则，不阻塞本地工作。来源可在 hcu-infer-pilot 中查找，不把其完整自动生成器或固定机型数据复制成本流程要求：

| 当前问题 | 来源（相对本文件） |
| --- | --- |
| 架构、roofline、优化优先级 | `kernel-hip-handbook/SKILL.md` |
| VGPR/SGPR、scratch、LDS 与并发限制 | `kernel-hip-handbook/references/occupancy.md` |
| 合并访问、向量化、LDS 布局与流水线 | `kernel-hip-handbook/references/memory-hierarchy.md` |
| wave 归约、shuffle 与分歧 | `kernel-hip-handbook/references/wavefront-optimization.md` |
| 粒度重构、原子、循环不变量 | `kernel-hip-handbook/references/launch-and-refactor.md` |
| shape 特化、保留 fallback | `kernel-hip-handbook/references/shape-specialization.md` |
| 选法、触发与跳过条件 | `kernel-hip-optimize/references/optimization_catalog.md`、`kernel-hip-optimize/references/method_registry.json` |
| profile 指标和 ISA 模式 | `kernel-hip-optimize/references/hcu_metrics_guide.md`、`kernel-hip-optimize/references/hcu_isa_signatures.json` |
| 分支、消融与方法状态 | `kernel-hip-optimize/references/optimizer-loop.md` |

记录实际读取的来源路径、提交或文件哈希、章节/方法 ID、适用 gfx 和影响的决策。方法 ID 只能从现有 registry 读取；不使用 registry 时以描述性假设标识，不能伪称机器验证通过。

手册中的“当前机器”、带宽/算力、bank 数、寄存器容量、最大 waves、占用率公式和典型加速比都是参考信息。优先核对目标设备与 DTK、代码对象资源、实际带宽及可用硬件资料。缺失规格时报告未知，不套用别的型号；示例中的文件或 CSV 不存在时不能据此编造参数。occupancy 是约束与诊断信号，不是越高越好的独立目标。

## 2. 从证据选择方法

先固定语义和 benchmark 契约。估算 FLOPs 与必要读写字节，并区分算法有效字节、缓存流量与实际 HBM 流量；有效带宽不是 HBM 计数器带宽。融合前后的带宽公式必须明确，不能减少统计字节或省略算子工作来宣称加速。roofline 结合目标设备带宽、dtype 对应算力以及 profile 判定，不能仅从算术强度推断实际瓶颈。

| 观测或瓶颈 | 优先候选 | 必须验证的条件 |
| --- | --- | --- |
| 接近可达带宽，重复全局读写或多次 launch | 融合、消除中间张量、epilogue 合并 | 中间值是否被别处使用；融合后的数值顺序、寄存器压力及整体调用时延 |
| 带宽不足、零散/标量访存 | lane/地址映射、合并访问、向量化、线程粗化 | 对齐、stride、向量尾部、alias 约束；访存指令与请求变化 |
| 有重复读取且可复用数据 | LDS tiling、布局/swizzle | 全局流量下降；LDS 用量、bank 冲突和同步；纯流式 kernel 不强塞 LDS |
| 大量原子竞争、过细粒度 | block-per-output、局部归约、可证明的 scatter→gather | 重复索引、初始化、边界与确定性语义；重构不能丢失贡献 |
| 小 group/行归约 | wave64/sub-wave shuffle、分层归约 | group 宽度与 lane 分组、尾部 mask、跨 wave 汇总和数值误差 |
| VGPR/SGPR/LDS 压力或 scratch spill | 缩短 live range、tile/unroll/block 调整、适量 launch bounds | 编译资源与实测 occupancy、spill/额外访存，不只看寄存器数 |
| 矩阵计算低效 | 已验证 CK Tile/HCU 路径、tile 与数据复用、MMAC | dtype/layout/架构支持、完整 global→LDS→matrix-read→MMAC→epilogue 路径 |
| load/compute 串行或等待停顿 | 双缓冲、software pipeline、独立指令交错 | 资源增量、消费者等待、barrier 一致性、SQTT/ISA 与时延 |
| 小 grid/launch 主导 | 合理粒度、batch/fusion、现有 persistent 路径 | CU 利用与总调用开销，不能仅比较 kernel 内计时 |
| 不同 shape 区间最优配置不同 | host dispatch、模板特化 | shape/dtype/layout/gfx 守卫，边界与未命中特化时的通用 fallback |

先做有证据的结构性优化，再考虑编译提示。`__restrict__` 要有不别名契约；循环展开/粗化不能忽略资源代价。FP16/BF16/FP8/TF32、fast-math 或累加精度变化只在既定语义与精度要求内选择，不为通过测试放宽容差。不引入未经当前 HCU 目标确认的 FP4 硬件路线。

## 3. 候选与验证循环

1. profile 当前最佳，记录瓶颈、证据强弱和可改善空间。选择一个主要假设，写出来源、触发指标、预期变化、跳过其它方法的理由和停止条件。
2. 保存父版本的源码/补丁及哈希、配置、构建产物标识。用独立候选编号探索 tile/vector width/block/LDS 等一个维度；不要求创建 Git 分支，也不授权并行 agent 或同卡并发性能运行。当前最佳作为 no-op 对照。
3. 构建并确认实际导入的新扩展。先做正确性，再同步计时；覆盖目标 shape、非整除尾部、支持的非连续输入、dtype/mode、特殊数值和对应族的语义陷阱。整数索引、top-k tie、重复 scatter 索引等按原契约判断。
4. 同卡、同输入、同统计方式复测基线/父版本/候选，必要时交错运行排除漂移。按 W 的目标指标和邻近回归门槛选冠军，不以单次最快值或非目标 shape 改善晋升。
5. 对正确候选收集配对 PMC；内存相关检查 read/write；资源问题检查 code-object；涉及指令机制的优化核验最终 ISA。调用内置 profiler 摘要确定下一步。
6. 对冠军做必要消融：固定 shape、launch/tile 参数、工具链和测量条件，仅移除待归因的方法。无法拆分的组合按组合归因；no-op 拷贝、未实际运行的消融、变更了其它超参数的比较均不能证明单个方法贡献。
7. keep/reject/inconclusive 决策后更新状态；失败分支保留日志并回到当前最佳。失败修复后仍需完整正确性与计时门控，不以无限重试代替诊断。

## 4. HCU ISA 和工具规则

- 对参考覆盖的 gfx936/gfx938 使用 wave64 设计；其它 gfx 先核实。重查 CUDA warp32、shuffle 宽度、mask、block 映射。不能因 wave 同步假设删除跨线程必要同步。
- 针对矩阵 kernel，优先工程已经支持且能编译的 CK Tile/HCU 示例；缺少依赖时保留已有 HIP 路线并记录约束，不自动引入新依赖。跨平台矩阵指令名称仅作类比，目标平台的 MMAC/MMOP 是否成立以编译产物为证。
- builtin 使用精确来源签名、类型/地址空间和 gfx 守卫，先做最小目标编译探针。`__has_builtin` 否定或仅编译成功都不是最终结论；不要根据指令助记符臆造 intrinsic。
- 使用目标构建产生并实际加载的 `.so`/code object，核对时间戳或哈希及目标符号，再运行 `hipprof --codeobj-analyze <binary>` 和 `dccobjdump --inputs=<binary> --show-sass --show-instruction-encoding --separate-functions`。资源/符号选项先查安装版本帮助。
- 向量化看相关 global/buffer/flat load/store 宽度；矩阵路径看目标 `v_mmac`/MMOP、matrix-load、DS matrix-read；LDS 路径检查 staging、读写与同步。只见到别的函数指令不算通过。
- 内联汇编是源码/CK Tile 路线无法表达且已有 ISA 证据时的最后选项。依据目标指令语义处理 `s_waitcnt` 和 `s_barrier`：核对 global 消费者、LDS/scalar 消费者以及跨线程可见性，不能盲删等待或把固定 wait 模板套到所有架构。
- PMC 不可用时保存准确错误并降级到计时、资源和 ISA；缺失指标不能写成零。SQTT 视工具支持与开销使用，导出依赖的 `llvm-objdump` 不替代最终 `dccobjdump` 核验。

分别记录候选和方法状态：构建/正确性失败是候选失败，不能证明思路无效；ISA 确认且消融收益超过噪声才称方法有效；ISA 确认但无贡献为 ineffective；相关完整 dump 明确缺少所声称指令为 implementation_failed；dump 不全/符号不匹配/工具缺失为 inconclusive。更快的正确候选若只是某项方法归因失败，可按整体性能及回归门槛保留，但撤销该方法的成功声明；若 ISA 是安全性或任务的必需验收条件，则不得据此绕过。

## 5. 状态、停止和交付

`.lightop-kernel-agent/state.json` 至少记录任务标识、K/R/W/E 摘要或计划路径、基线与当前最佳源码哈希/产物、当前轮次、有效噪声阈值、预算、候选结果、方法归因、证据路径和下一步。在既有尝试台账的每轮条目增加来源/方法、预期指标、实测 profile-diff、ISA 状态和消融结论。恢复时核验状态与源码是否一致。

保留完整工作流中第二次正确候选提升不足 5% 的深度诊断门槛；5% 是触发调查的条件，不替代本任务的噪声阈值。连续三轮未超过噪声带，下一次编辑前做 SQTT/资源/ISA 诊断（不可用则记录），并重新判断剩余空间。没有新证据支持的方向时停止，不能循环重复微调。

满足性能目标且最终回归通过则报告达标；预算耗尽、环境不可用、未解正确性故障、目标不可达或平台期无新方向时，报告停止原因、当前最佳与目标差距、证据限制及下一项具体行动。不得将停止等同于性能达标。

## 6. 可选复用 optimizer 脚本

本流程默认由 agent 在 LightOp 源码和现有测试中直接执行。仅在确实需要独立实验、且已满足 `kernel-hip-optimize` 的 baseline/reference/dims/指针尺寸/`solve(...)` 契约时，读取其入口并复用脚本；manifest 未 ready 或字段缺失时先补齐，不能虚构契约。自动生成器与复杂族指导仅为候选，不是正确性证明。

独立探针及生成的 `.hip` 留在临时证据目录，最终 LightOp 实现仍为 `.cu`，沿用已有 wrapper/绑定和正式测试。独立 benchmark、自动 scaffold、no-op 冠军或 optimizer 的 finalize 成功都不能替代 LightOp 安装、导入、正确性及性能回归。
