---
name: hcu-profiler-report
description: LightOp 的 HCU/ROCm 算子需要性能分析证据时使用：运行 hipprof 或更深入的 ROCm/DTK profiler，采集同步 benchmark 日志，总结 API/kernel/内存耗时，检查可用计数器或目标 ISA，并给出且仅给出一个具体的下一步算子修改。
---

# HCU 性能分析报告

benchmark 数字不足以确定下一步 LightOp HIP/ROCm kernel、绑定、dispatch 或配置表修改，需要依据 HCU 性能分析证据时，使用本 Skill。

规则是：

```text
先采集性能证据，再诊断，最后优化
```

输出不能只写“受内存限制”或“启动开销”，而应给出从实测 HCU 证据，到可能机制，再到一个可执行 LightOp 编辑的推理链。

## 官方参考

原生 profiler 的命令、指标和支持范围以目标环境安装的 DTK 版本及工具帮助为准。公开源码与测试可参考 [HYGON-AI/flash-attention-das](https://github.com/HYGON-AI/flash-attention-das)；框架层 trace 范围归因可参考 [HYGON-AI/SkillHub 的 torch-trace-operator-profiler](https://github.com/HYGON-AI/SkillHub/tree/130aa48c2c55ce3366e01daf54e8494b80d78138/skills/torch-trace-operator-profiler)。两类证据分别核验，不能互相替代。

优先 DTK/HCU 工具。首次使用 `hipprof` 测量 API/kernel/memcpy 时间。只有耗时证据无法解释下一次修改时，才升级到 `rocprof`、`rocprofv3`、`rocprof-compute`、计数器工具，或目标 ISA/code-object 检查。

## 调用时机

出现以下任一情况，调用 `hcu-profiler-report`：

- 一个通过正确性的 LightOp 优化候选刚完成 benchmark，闭环需要在下次编辑前解释必需的每候选 `hipprof --pmc`。
- LightOp 基线 benchmark 已通过，但尚无基线 profile 摘要。
- 正确候选相对基线或先前最佳处于 +/-2% 范围。
- 第二次通过正确性的优化尝试，相对父版本或基线提升不足 5%，下次优化编辑前需要深度分析。
- 正确候选在一个或多个重要 shape 上回退。
- 候选明显快于预期，需要解释。
- benchmark 结果仍无法明确下一次编辑。
- 性能复核者要求 profile 证据。
- 候选可能受限于启动开销、主机与设备复制、全局内存布局、LDS/bank conflict、VGPR/occupancy 压力、wavefront 分歧、MFMA 利用不足或 shape-dispatch 选择。

正确性失败时，不进行性能分析，除非故障本身是 profiler 采集问题。先修复正确性。

## 设备选择门槛

任何 benchmark 或性能采集前，在将要运行工作负载的同一宿主/容器环境记录当前卡状态。

- 运行 `hy-smi` 或 `rocm-smi`。
- 选择 HCU 利用率低、显存占用低的卡。
- benchmark 和 profiling 命令使用 `HIP_VISIBLE_DEVICES=<idle-card>`。
- 基线/候选配对采集保持同一张卡。
- 将状态输出和所选卡保存到产物目录及摘要。工具不可用时记录失败命令，除非用户接受这一限制，否则性能结果不能作为最终证据。
- 首次优化 profile 或基线比较前，纳入 `.lightop-kernel-agent/device-bandwidth.txt` 中所选卡的设备带宽校准。缺失时，在同一环境校准并记录实际读、写、复制/读写及 triad 带宽，再解释算子带宽。

## 必需产物

产物保存在用户指定证据位置或被忽略的 LightOp 闭环状态中，留在交付源码 diff 之外。不要向算子树添加 benchmark 结果或 Markdown 报告。复用 benchmark 入口；耗时 profiling/性能任务不在测试 import 或默认 pytest 时运行。以下示例假定所选 benchmark 显式支持其选项：

```text
.lightop-kernel-agent/device-bandwidth.txt
.lightop-kernel-agent/profile-artifacts/<version>/
  device-status.txt           # benchmark/profile 前的 hy-smi 或 rocm-smi
  benchmark.log
  hipprof.txt
  hipprof-pmc-all/            # 每个通过正确性的优化候选都必须提供
  sqtt-json/                  # SQTT 可用时必须提供
  rocprof.csv                 # 使用 rocprof/rocprofv3 时提供
  rocprof-stats.csv           # 可用时提供
  rocprof-compute/            # 可用时提供
  code-object-metadata.txt    # 已提取时提供
  hcu-isa.txt                 # 已提取时提供
  resource-usage.txt          # VGPR/SGPR/LDS/occupancy/code-object 证据
  digest.md
```

比较候选时，摘要必须给出基线或父版本的产物路径。

## 工作流

1. 先选一个代表性 shape。优先能暴露回退、平台期、启动开销或疑似瓶颈的最小 shape。
2. 建立聚焦 benchmark harness，固定预热、dtype、shape、seed，并在计时区间前后显式同步。
3. 执行设备选择门槛，用 `HIP_VISIBLE_DEVICES` 固定选定的空闲卡。
4. 首次优化编辑或基线 profile 前，运行或引用所选卡的设备带宽校准，将算子有效带宽目标与实测读/写/复制上限比较。
5. profiling 前先采集普通 benchmark 输出，避免把 profiler 开销计入性能结论。
6. 使用 `hipprof` 完成首次 API/kernel/memcpy 耗时测量。
7. 每个通过正确性的优化候选，使用支持的 `--pmc`、`--pmc-read`、`--pmc-write` 等形式收集 `hipprof` PMC 证据。选择下一次编辑前，解释缓存行为、内存/缓存流量、LDS 或 bank-conflict 证据和 occupancy/资源压力。计数器组或 occupancy 信号不可用时，记录准确尝试命令和错误输出。
8. 深度分析门槛触发时，在选择下一次编辑前补充可用的 SQTT JSON、`dccobjdump` 反汇编、code-object 资源用量和 LDS/寄存器/occupancy 证据。
9. `hipprof` 显示一个或少数主导 kernel，但无法解释原因时，用已安装的 ROCm/DTK profiler 采集更深的计数器证据。
10. 问题看起来对代码生成敏感时，编辑前检查目标 ISA 或 code-object 元数据。
11. 将候选与基线或父版本比较，不只看绝对指标。
12. 依据 [metrics.md](references/metrics.md) 的指标分组诊断。
13. 按下方模板写 `digest.md`，最后一节必须且只能包含一个下一步编辑。
14. 在 LightOp 闭环台账更新摘要路径、瓶颈类别和选定的下一次编辑。

## 常用命令

可直接复制的命令形式见 [examples.md](references/examples.md)。

从 LightOp 根目录进行最小首次采集。`<agent-loop-root>` 指外层 `.agents/skills/lightop-kernel-agent-loop`，共用该目录唯一的带宽校准脚本；本 profiler 子目录不另存副本：

```bash
bash <agent-loop-root>/scripts/measure-device-bandwidth.sh \
  --docker <container> \
  --workdir <container-lightop> \
  --hip-visible-devices <idle-card> \
  --output .lightop-kernel-agent/device-bandwidth.txt
mkdir -p .lightop-kernel-agent/profile-artifacts/v000_baseline
hy-smi 2>&1 | tee .lightop-kernel-agent/profile-artifacts/v000_baseline/device-status.txt || \
  rocm-smi 2>&1 | tee .lightop-kernel-agent/profile-artifacts/v000_baseline/device-status.txt
HIP_VISIBLE_DEVICES=<idle-card> python test/ci_tests/test_<op>.py --benchmark 2>&1 | tee .lightop-kernel-agent/profile-artifacts/v000_baseline/benchmark.log
HIP_VISIBLE_DEVICES=<idle-card> hipprof python test/ci_tests/test_<op>.py --benchmark 2>&1 | tee .lightop-kernel-agent/profile-artifacts/v000_baseline/hipprof.txt
```

使用 benchmark 脚本时：

```bash
mkdir -p .lightop-kernel-agent/profile-artifacts/v001_candidate
hy-smi 2>&1 | tee .lightop-kernel-agent/profile-artifacts/v001_candidate/device-status.txt || \
  rocm-smi 2>&1 | tee .lightop-kernel-agent/profile-artifacts/v001_candidate/device-status.txt
HIP_VISIBLE_DEVICES=<idle-card> hipprof python test/<family>/benchmark_<op>.py 2>&1 \
  | tee .lightop-kernel-agent/profile-artifacts/v001_candidate/hipprof.txt
mkdir -p .lightop-kernel-agent/profile-artifacts/v001_candidate/hipprof-pmc-all
HIP_VISIBLE_DEVICES=<idle-card> hipprof --pmc --pmc-type 3 \
  -o .lightop-kernel-agent/profile-artifacts/v001_candidate/hipprof-pmc-all/pmc \
  python test/<family>/benchmark_<op>.py
HIP_VISIBLE_DEVICES=<idle-card> hipprof --pmc-read --pmc-type 3 \
  -o .lightop-kernel-agent/profile-artifacts/v001_candidate/hipprof-pmc-all/pmc-read \
  python test/<family>/benchmark_<op>.py
HIP_VISIBLE_DEVICES=<idle-card> hipprof --pmc-write --pmc-type 3 \
  -o .lightop-kernel-agent/profile-artifacts/v001_candidate/hipprof-pmc-all/pmc-write \
  python test/<family>/benchmark_<op>.py
```

第二次通过正确性的优化提升不足 5% 后，必须进行深度采集：

```bash
mkdir -p .lightop-kernel-agent/profile-artifacts/v002_deep/{hipprof-pmc-all,sqtt-json}
hy-smi 2>&1 | tee .lightop-kernel-agent/profile-artifacts/v002_deep/device-status.txt || \
  rocm-smi 2>&1 | tee .lightop-kernel-agent/profile-artifacts/v002_deep/device-status.txt
HIP_VISIBLE_DEVICES=<idle-card> hipprof --pmc --pmc-type 3 \
  -o .lightop-kernel-agent/profile-artifacts/v002_deep/hipprof-pmc-all/pmc \
  python test/<family>/benchmark_<op>.py
HIP_VISIBLE_DEVICES=<idle-card> hipprof --pmc-read --pmc-type 3 \
  -o .lightop-kernel-agent/profile-artifacts/v002_deep/hipprof-pmc-all/pmc-read \
  python test/<family>/benchmark_<op>.py
HIP_VISIBLE_DEVICES=<idle-card> hipprof --pmc-write --pmc-type 3 \
  -o .lightop-kernel-agent/profile-artifacts/v002_deep/hipprof-pmc-all/pmc-write \
  python test/<family>/benchmark_<op>.py
HIP_VISIBLE_DEVICES=<idle-card> hipprof --sqtt --sqtt-type 1 --output-type 0 \
  -d .lightop-kernel-agent/profile-artifacts/v002_deep/sqtt-json/ \
  python test/<family>/benchmark_<op>.py
dccobjdump --inputs=<binary-or-so> --show-sass --show-instruction-encoding \
  --separate-functions > .lightop-kernel-agent/profile-artifacts/v002_deep/hcu-isa.txt
hipprof --codeobj-analyze <binary-or-so> \
  > .lightop-kernel-agent/profile-artifacts/v002_deep/resource-usage.txt
```

将 `<binary-or-so>` 替换为实际构建产生的编译扩展、code object 或提取的 kernel 二进制。已安装 DTK 不支持某条命令时，将输出或错误保留在产物目录，并在摘要说明缺失证据。

可选辅助工具：

```bash
python3 <skill-root>/scripts/hcu_profile_digest.py \
  --hipprof .lightop-kernel-agent/profile-artifacts/v001_candidate/hipprof.txt \
  --baseline-hipprof .lightop-kernel-agent/profile-artifacts/v000_baseline/hipprof.txt \
  --benchmark-log .lightop-kernel-agent/profile-artifacts/v001_candidate/benchmark.log \
  --kernel-regex "<kernel-name-pattern>" \
  --output .lightop-kernel-agent/profile-artifacts/v001_candidate/digest.md
```

辅助工具只做初步分析。将下一次编辑作为最终选择前，仍需检查 profiler 输出以及计数器/ISA 证据。

## 应检查的指标

先检查以下分组，完整列表和解释规则见 [metrics.md](references/metrics.md)：

- 运行身份：LightOp commit、算子、公共 API、shape、dtype、gfx 架构、DTK/ROCm/PyTorch 版本和构建命令。
- 设备选择：`hy-smi` 或 `rocm-smi` 输出、选定的 `HIP_VISIBLE_DEVICES`、HCU 利用率和显存占用。
- benchmark 计时：预热、重复、p50/p90/mean、同步点、方差、基线与候选变化量。
- `hipprof` 耗时：HIP API 总时间、kernel 时间、memcpy/memset 时间、启动开销，以及按 inclusive/exclusive 时间排序的热点 kernel。
- kernel 启动形状：grid/block、每 block 的 wave 数、occupancy 线索、block 数与 CU 数之比、微小 kernel 开销。
- 内存路径：全局 load/store 流量、合并访存、对齐/向量宽度、L2/HBM 压力、冗余读写和临时张量。
- LDS 路径：LDS 使用、bank conflict、barrier 和共享暂存成本。
- 计算路径：MFMA/向量 ALU 利用率、转换/量化开销、标量控制开销、epilogue 融合成本。
- 资源压力：VGPR/SGPR、spill/scratch、每 block LDS、occupancy 限制项和过度展开。
- Dispatch/config：错误 shape 分支、过宽配置、缺失 gfx 特化或误选 fallback。
- ISA/code object：热点指令窗口、向量宽度、MFMA 选择、过度标量化及 scratch load/store。

第二次优化 <5% 的强制门槛要求摘要包含全部分节：`hipprof` PMC all、SQTT JSON 或工具不可用原因、`dccobjdump` 反汇编、code-object 资源用量，以及明确的 LDS/寄存器/occupancy 解释。

在目标机器核实具体工具是否可用：

```bash
which hipprof || true
which rocprof || true
which rocprofv3 || true
which rocprof-compute || true
hipcc --version
```

## 摘要模板

```markdown
### HCU 性能分析摘要：<operator> @ <version>

环境
- LightOp 根目录：
- 仓库 commit：
- GPU / gfx：
- 设备状态命令：
- 选定的 HIP_VISIBLE_DEVICES：
- 运行前的 HCU 利用率 / 显存：
- DTK / ROCm / PyTorch：
- 构建命令：
- Benchmark 命令：
- Shape / dtype：
- 基线或父版本 profile：
- 候选 profile：
- 产物：

核心结论
- 瓶颈类别：
- 主导耗时或热点：
- 置信度：High | Medium | Low
- 本次 profile 有效的依据：

证据
| 信号 | 基线 | 候选 | 变化量 | 解释 |
|---|---:|---:|---:|---|
| <指标/日志行> | <值> | <值> | <变化量> | <含义> |

Profiler 热点
- <kernel/API/复制/config 分支>：<实测信号> -> <含义>

深度分析产物
- hipprof PMC all：
- SQTT JSON：
- dccobjdump：
- code-object 资源用量：
- LDS/寄存器/occupancy：

计数器 / ISA 分析
- 计数器来源：
- 热点指令窗口：
- 疑似代码生成或资源问题：
- 指令级或 launch 级编辑：

推理链
1. 实测信号：
2. 可能机制：
3. 其他假设较弱的原因：
4. 风险：
5. 编辑后的验证命令：

下一次编辑
- <且仅一个具体的 LightOp 代码、绑定、配置或 benchmark 编辑>
```

## 诊断提示

- HIP API 时间高、kernel 时间很短：减少 Python/C++ launch 次数，融合相邻算子，或转到已有的融合 LightOp 路径。
- memcpy/memset 时间高：移除主机与设备传输，避免临时物化，或明确输出分配与复用。
- 单一 kernel 主导且 benchmark 扩展性差：检查内存布局、向量化 load/store、block 大小和 dispatch 配置。
- occupancy 偏低或出现 spill：降低 VGPR 压力、拆分 kernel、减少展开或针对目标 shape 特化。
- LDS/barrier 症状明显：减少暂存、修复 bank conflict，或简化 wave 间同步。
- 具备 MFMA 能力的 GEMM/MoE 路径性能差：确认命中了预期 asm/rocBLAS/hipBLASLt/CK/Triton 的目标平台路线，shape 配置匹配 `W`。
- 候选仅在一个 shape 获胜：记录到性能映射，并增加 shape/gfx dispatch，不替换更广泛的默认路径。

## 台账规则

- 在尝试台账中保留 `hipprof.txt`、benchmark 日志、更深 profiler 导出和摘要路径。
- 只有正确性通过且 benchmark 显示实测改善后，才将 profile 驱动的修改提升到优化台账。
- profiler 工具不可用时，记录尝试命令、失败及采用的更低成本替代证据。

## 导入来源的元数据

导入时保留了以下来源专用 frontmatter，但它不属于可移植 Agent Skills 的 frontmatter 契约。

```yaml
type: flow
```
