---
name: build-hcu-wheel
description: 使用 scripts/build_hcu_wheel.sh 编译 LightOp 并产出 manylinux wheel。需要编译 LightOp、生成发布用 wheel、或在本地/CI 复现编译时使用。
---

# 编译 LightOp manylinux wheel

`scripts/build_hcu_wheel.sh` 是统一的编译入口：本地编译、`pr-test` 与 `release-pypi` 都通过它生成 wheel。

## 前置环境

脚本只检查以下前置依赖，缺失会直接报错退出：

| 依赖 | 要求 | 检查方式 |
| --- | --- | --- |
| Python | 环境中的 `python3`，3.9+，带 pip | `python3` 版本与 `python3 -m pip` |
| DTK | 含 `.info/rocm_version` 与 `hipcc` | `ROCM_HOME`/`ROCM_PATH`，否则 `/opt/dtk`、`/opt/rocm` |
| PyTorch | HIP 版（`torch.version.hip` 非空） | 用环境中的 `python3` 导入 torch |
| AICC | `hcu_llvm_installer.sh` 安装到 DTK 目录，提供 `bin/aicc` | `<DTK>/bin/aicc` 或 PATH 中的 `aicc` |

- 脚本会自动 `source <DTK>/env.sh`，无需手动加载 DTK 环境。
- 找到 DTK 后会把 `ROCM_HOME` 指向它（未显式设置时），保证 `setup.py` 读取一致。
- `ninja`、`wheel`、`setuptools`、`packaging`、`auditwheel`、`patchelf` 由脚本安装，走环境中的 pip 配置。

缺少 AICC 时先安装（内网资源示例）：

```bash
wget "${RESOURCE_SERVER_URL}/ai_cc/Nightly/hcu_llvm_installer.sh"
bash hcu_llvm_installer.sh --dtk-dir /opt/dtk
```

## 基本用法

在仓库根目录执行：

```bash
./scripts/build_hcu_wheel.sh
```

默认行为：

1. 检查 DTK、torch、AICC；
2. 安装编译依赖；
3. `python3 setup.py bdist_wheel` 编译；
4. 调用 `.github/scripts/manylinux_repair.sh` 执行 `auditwheel repair`；
5. 输出 manylinux wheel 到 `dist-manylinux/`。

### 常用示例

指定编译并行度：

```bash
./scripts/build_hcu_wheel.sh --jobs 32
```

输出到自定义目录：

```bash
./scripts/build_hcu_wheel.sh -o /tmp/lightop-wheel
```

脚本固定使用环境中的 `python3`（如使用虚拟环境，先 `source .../bin/activate`，保证 `python3` 指向该环境）。

仅编译普通 wheel（不 repair，用于排查编译问题）：

```bash
./scripts/build_hcu_wheel.sh --no-repair
```

## 选项

| 选项 | 说明 |
| --- | --- |
| `-o, --output-dir DIR` | 输出目录；默认 `dist-manylinux`（`--no-repair` 时默认 `dist`） |
| `--jobs N` | 编译并行度，等价 `MAX_JOBS=N`；默认交给 ninja |
| `--clean` | 编译前删除仓库 `build/` 目录 |
| `--no-repair` | 跳过 auditwheel repair，仅调试用 |
| `-h, --help` | 显示帮助 |

## 产物

- `dist-manylinux/lightop-<version>-cp<xy>-cp<xy>-manylinux_<glibc>_x86_64.whl`
- 平台标签由构建机 glibc 决定，例如 rockylinux8.6 下为 `manylinux_2_28_x86_64`。
- HIP/ROCm、torch、ffmpeg 等运行时库由目标环境提供，不会打进 wheel（见 repair 脚本的 `excludes`）。

## CI 中的用法

`pr-test` 与 `release-pypi` 的编译步骤均为：

```bash
# 前置: 已在环境中安装 HIP 版 torch
bash scripts/build_hcu_wheel.sh --output-dir dist-manylinux
```

- `pr-test` 随后用 `pip install --force-reinstall --no-deps dist-manylinux/*.whl` 安装并测试该 manylinux wheel，确保测试对象与发布产物一致。
- `release-pypi` 直接上传 `dist-manylinux/*.whl`。

## 排错

- **未检测到 DTK**：设置 `ROCM_HOME`/`ROCM_PATH`，或确认 DTK 位于 `/opt/dtk`。
- **torch 不是 HIP 版**：安装与 DTK 匹配的 HIP 版 PyTorch，勿使用 CUDA 版。
- **未检测到 AICC**：确认安装到了当前 DTK 目录，且存在 `bin/aicc`。
- **auditwheel 报找不到依赖库**：确认已 `source <DTK>/env.sh`（脚本会做），并检查 `LD_LIBRARY_PATH` 是否覆盖 torch/DTK 库目录。
