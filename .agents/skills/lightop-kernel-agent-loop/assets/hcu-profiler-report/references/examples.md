# hcu-profiler-report 示例

未特别说明时，从 LightOp 根目录运行命令。

## 环境探测

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("device:", torch.cuda.get_device_name(0))
print("gcn:", torch.cuda.get_device_properties(0).gcnArchName)
PY
hipcc --version
which hipprof || true
which rocprof || true
which rocprofv3 || true
which rocprof-compute || true
```

## 首次 hipprof 采集

```bash
version=v000_baseline
mkdir -p ".lightop-kernel-agent/profile-artifacts/$version"
python test/ci_tests/test_<op>.py --benchmark 2>&1 \
  | tee ".lightop-kernel-agent/profile-artifacts/$version/benchmark.log"
hipprof python test/ci_tests/test_<op>.py --benchmark 2>&1 \
  | tee ".lightop-kernel-agent/profile-artifacts/$version/hipprof.txt"
```

## 候选比较

```bash
version=v001_candidate
mkdir -p ".lightop-kernel-agent/profile-artifacts/$version"
python test/<family>/benchmark_<op>.py 2>&1 \
  | tee ".lightop-kernel-agent/profile-artifacts/$version/benchmark.log"
hipprof python test/<family>/benchmark_<op>.py 2>&1 \
  | tee ".lightop-kernel-agent/profile-artifacts/$version/hipprof.txt"
python3 <skill-root>/scripts/hcu_profile_digest.py \
  --hipprof ".lightop-kernel-agent/profile-artifacts/$version/hipprof.txt" \
  --baseline-hipprof ".lightop-kernel-agent/profile-artifacts/v000_baseline/hipprof.txt" \
  --benchmark-log ".lightop-kernel-agent/profile-artifacts/$version/benchmark.log" \
  --kernel-regex "<kernel-name-pattern>" \
  --operator "<lightop-op>" \
  --version "$version" \
  --output ".lightop-kernel-agent/profile-artifacts/$version/digest.md"
```

## 升级到 rocprof / rocprofv3

`hipprof` 已定位热点 kernel，但尚不能解释机制时使用。

```bash
version=v002_counters
mkdir -p ".lightop-kernel-agent/profile-artifacts/$version"
rocprof --stats -o ".lightop-kernel-agent/profile-artifacts/$version/rocprof.csv" \
  python test/<family>/benchmark_<op>.py
```

目标系统提供 `rocprofv3` 时：

```bash
version=v002_trace
mkdir -p ".lightop-kernel-agent/profile-artifacts/$version"
rocprofv3 --kernel-trace --stats --output-format csv \
  --output-file ".lightop-kernel-agent/profile-artifacts/$version/rocprofv3" \
  -- python test/<family>/benchmark_<op>.py
```

## 升级到 rocprof-compute

需要 kernel 级计数器时使用。

```bash
version=v003_compute
mkdir -p ".lightop-kernel-agent/profile-artifacts/$version"
rocprof-compute profile -n "$version" --path ".lightop-kernel-agent/profile-artifacts/$version" \
  -- python test/<family>/benchmark_<op>.py
rocprof-compute analyze -p ".lightop-kernel-agent/profile-artifacts/$version/$version" \
  | tee ".lightop-kernel-agent/profile-artifacts/$version/rocprof-compute.txt"
```

## 同步模式

临时 benchmark 代码使用显式同步：

```python
for _ in range(warmup):
    op(*args)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(repeats):
    op(*args)
end.record()
torch.cuda.synchronize()
latency_ms = start.elapsed_time(end) / repeats
```

## 摘要检查清单

每份摘要应回答：

- 分析了哪个算子、wrapper、绑定和 kernel 路径？
- 选择了哪个 shape/dtype/gfx 配置？
- 相对基线或父版本改变了什么？
- 主导成本是 HIP API launch、memcpy/memset、kernel 执行、配置分发、内存路径、计算路径、资源压力还是代码生成？
- 下一步应该尝试哪一个编辑？
