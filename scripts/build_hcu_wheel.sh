#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# 编译 LightOp 并直接产出 manylinux wheel。
#
# 前置环境（脚本检查，缺失直接报错）：
#   - Python 3.9+ 与 pip
#   - DTK（默认 /opt/dtk，需包含 .info/rocm_version 与 hipcc）
#   - HIP 版 PyTorch
#   - AICC（hcu_llvm_installer.sh 安装到 DTK 目录，提供 bin/aicc）
# 其余编译依赖（ninja、wheel、setuptools、packaging、auditwheel、patchelf）
# 由本脚本安装。

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR=""
PYTHON_BIN="python3"
JOBS=""
CLEAN_BUILD=0
REPAIR=1

usage() {
    cat <<'EOF'
编译 LightOp 并直接产出 manylinux wheel。

前置环境（脚本检查，缺失直接报错）：
  - Python 3.9+ 与 pip
  - DTK（默认 /opt/dtk，需包含 .info/rocm_version 与 hipcc）
  - HIP 版 PyTorch
  - AICC（hcu_llvm_installer.sh 安装到 DTK 目录，提供 bin/aicc）

用法: ./scripts/build_hcu_wheel.sh [选项]

选项:
  -o, --output-dir DIR     wheel 输出目录 (默认: 生成 manylinux 时为 dist-manylinux,
                           使用 --no-repair 时为 dist)
      --jobs N             编译并行度, 等价于设置 MAX_JOBS (默认交给 ninja 决定)
      --clean              编译前清理 build/ 目录
      --no-repair          只生成普通 wheel, 不执行 auditwheel repair (仅调试用)
  -h, --help               显示帮助
EOF
}

log() {
    echo "[build_hcu_wheel] $*"
}

die() {
    echo "[build_hcu_wheel] 错误: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir)
            [[ $# -ge 2 ]] || die "$1 缺少参数"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --jobs)
            [[ $# -ge 2 ]] || die "$1 缺少参数"
            JOBS="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --no-repair)
            REPAIR=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "未知参数: $1 (使用 --help 查看用法)"
            ;;
    esac
done

if [[ -n "$JOBS" && ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    die "--jobs 需要正整数, 得到: $JOBS"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ "$REPAIR" -eq 1 ]]; then
        OUTPUT_DIR="$REPO_ROOT/dist-manylinux"
    else
        OUTPUT_DIR="$REPO_ROOT/dist"
    fi
fi

# DTK 的 env.sh 会引用未初始化变量, 在 set -u 之前加载
set +u
for _dtk in "${ROCM_HOME:-}" "${ROCM_PATH:-}" "${DTK_DIR:-}" /opt/dtk /opt/rocm; do
    if [[ -n "$_dtk" && -f "$_dtk/env.sh" ]]; then
        # shellcheck disable=SC1090
        source "$_dtk/env.sh"
        break
    fi
done
set -u

DTK_ROOT="${ROCM_HOME:-${ROCM_PATH:-${DTK_DIR:-}}}"
if [[ -z "$DTK_ROOT" || ! -d "$DTK_ROOT" ]]; then
    DTK_ROOT=""
    for _dtk in /opt/dtk /opt/rocm; do
        if [[ -f "$_dtk/.info/rocm_version" ]]; then
            DTK_ROOT="$_dtk"
            break
        fi
    done
fi
[[ -n "$DTK_ROOT" ]] || die "未检测到 DTK: 请设置 ROCM_HOME/ROCM_PATH, 或安装到 /opt/dtk"

if [[ ! -f "$DTK_ROOT/.info/rocm_version" ]]; then
    die "DTK 目录 $DTK_ROOT 缺少 .info/rocm_version, 不是有效的 DTK 安装"
fi
if [[ ! -x "$DTK_ROOT/bin/hipcc" ]] && ! command -v hipcc >/dev/null 2>&1; then
    die "未找到 hipcc (DTK): 请 source $DTK_ROOT/env.sh 或检查 DTK 安装"
fi

DTK_ROOT="$(cd "$DTK_ROOT" && pwd)"

# 让 setup.py 中 torch 解析到的 ROCM_HOME 与脚本检测结果一致
if [[ -z "${ROCM_HOME:-}" ]]; then
    export ROCM_HOME="$DTK_ROOT"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "未找到 Python 解释器: $PYTHON_BIN"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"

if ! "$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 9):
    print(f"需要 Python 3.9+, 当前为 {sys.version.split()[0]}", file=sys.stderr)
    sys.exit(1)
PY
then
    die "Python 版本检查失败: $PYTHON_BIN"
fi

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    die "Python 解释器缺少 pip: $PYTHON_BIN"
fi

if ! "$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except Exception as exc:  # noqa: BLE001
    print(f"无法导入 torch: {exc}", file=sys.stderr)
    sys.exit(1)

if torch.version.hip is None:
    print("当前 torch 不是 HIP 版, 请安装与 DTK 匹配的 PyTorch", file=sys.stderr)
    sys.exit(1)

print(f"torch {torch.__version__} (HIP {torch.version.hip})")
PY
then
    die "PyTorch 前置检查失败"
fi

AICC_BIN=""
if [[ -x "$DTK_ROOT/bin/aicc" ]]; then
    AICC_BIN="$DTK_ROOT/bin/aicc"
elif command -v aicc >/dev/null 2>&1; then
    AICC_BIN="$(command -v aicc)"
fi
[[ -n "$AICC_BIN" ]] || die "未检测到 AICC: 请先执行 hcu_llvm_installer.sh --dtk-dir $DTK_ROOT (参见 CI 配置)"

DTK_VERSION="$(<"$DTK_ROOT/.info/rocm_version")"
log "DTK:  $DTK_ROOT ($DTK_VERSION)"
log "AICC: $AICC_BIN"

pip_install() {
    "$PYTHON_BIN" -m pip install --disable-pip-version-check "$@"
}

log "安装编译依赖..."
pip_install ninja wheel setuptools packaging
if [[ "$REPAIR" -eq 1 ]]; then
    pip_install auditwheel patchelf
fi

if [[ "$CLEAN_BUILD" -eq 1 ]]; then
    log "清理 $REPO_ROOT/build"
    rm -rf "$REPO_ROOT/build"
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

WHEEL_HOUSE="$(mktemp -d "${TMPDIR:-/tmp}/lightop-wheel.XXXXXX")"
cleanup() {
    rm -rf "$WHEEL_HOUSE"
}
trap cleanup EXIT

if [[ -n "$JOBS" ]]; then
    export MAX_JOBS="$JOBS"
fi

log "编译 wheel (MAX_JOBS=${MAX_JOBS:-auto})..."
cd "$REPO_ROOT"
if ! "$PYTHON_BIN" setup.py bdist_wheel --dist-dir "$WHEEL_HOUSE"; then
    die "wheel 编译失败"
fi

shopt -s nullglob
raw_wheels=("$WHEEL_HOUSE"/*.whl)
shopt -u nullglob
[[ ${#raw_wheels[@]} -eq 1 ]] || die "预期生成 1 个 wheel, 实际生成 ${#raw_wheels[@]} 个"

if [[ "$REPAIR" -eq 1 ]]; then
    log "生成 manylinux wheel..."
    bash "$REPO_ROOT/.github/scripts/manylinux_repair.sh" \
        "$WHEEL_HOUSE" "$OUTPUT_DIR"
else
    log "跳过 manylinux repair"
    cp "${raw_wheels[0]}" "$OUTPUT_DIR/"
fi

shopt -s nullglob
out_wheels=("$OUTPUT_DIR"/*.whl)
shopt -u nullglob
[[ ${#out_wheels[@]} -gt 0 ]] || die "未生成 wheel: $OUTPUT_DIR"

log "完成, 产物:"
for wheel in "${out_wheels[@]}"; do
    log "  $wheel"
done
