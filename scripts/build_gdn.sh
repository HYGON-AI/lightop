#!/usr/bin/env bash
# Build the GDN recurrent pybind module as a prebuilt artifact under lib/.
#
# aicc is required: the tuned -mllvm flags are aicc backend options and its
# codegen keeps the kernel bit-exact. hipcc rejects those flags (prints the
# option help, exits 0 and leaves an empty .so) and even a plain hipcc build
# changes the numerics, which drops the speculative accept rate 0.95 -> 0.89
# and costs ~8% end to end.
#
# setup.py is not involved on purpose: the wheel's BuildPyWithResources copies
# the repository lib/ tree into lightop/lib unchanged, so this script only
# produces lib/gdn/<arch>/_gdn<suffix> and asserts the kernel exports are
# present. The per-arch directory keeps a wheel built for several GPUs from
# loading another architecture's kernel.
#
# usage: scripts/build_gdn.sh [out.so]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SUFFIX="$(python3-config --extension-suffix)"
ARCH="${GDN_ARCH:-gfx938}"
OUT="${1:-$ROOT/lib/gdn/$ARCH/_gdn$SUFFIX}"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$(dirname "$OUT")"

AICC="${AICC:-/opt/dtk/bin/aicc}"
CXX_BIN="${CXX:-g++}"

GDN_FLAGS=(
  -O3 -fPIC "--offload-arch=$ARCH"
  -mllvm=-enable-latency-hack=true
  -mllvm=-mmac-latency=5
  -mllvm=-ds-load-store-latency=6
  -mllvm=-disable-machine-sink=True
  -mllvm=-check-valu-data-forward-hazards=0
  -mllvm=-disable-cluster-lds-memops=true
  -mllvm=-amdgpu-disable-backoff-barrier=false
  -mllvm=-support-768-vgprs=true
  -mllvm=-hcu-update-wait-by-reverse-search=true
  -mllvm=-hcu-pre-emit-load-store-opt=false
)

TORCH_INC=$(python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I'+p for p in include_paths()))")
TORCH_LIB=$(python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join('-L'+p for p in library_paths()))")
PY_INC=$(python3-config --includes)
ABI=$(python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

"$AICC" "${GDN_FLAGS[@]}" -I"$ROOT/csrc" \
    -c "$ROOT/csrc/gdn/gdn_recurrent.hip" -o "$BUILD/gdn_recurrent.o"

"$CXX_BIN" -O3 -fPIC -std=c++17 -w $PY_INC $TORCH_INC \
    -I/opt/dtk/include -I"$ROOT/csrc" -D__HIP_PLATFORM_AMD__=1 \
    -DTORCH_EXTENSION_NAME=_gdn -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
    -c "$ROOT/csrc/gdn/gdn_binding.cpp" -o "$BUILD/gdn_binding.o"

"$CXX_BIN" -shared "$BUILD/gdn_binding.o" "$BUILD/gdn_recurrent.o" \
    -L/opt/dtk/lib -lamdhip64 $TORCH_LIB -ltorch_python -ltorch -lc10 \
    -o "$OUT"

# A silent compile failure would ship an empty module; assert the entries.
for name in gdn_recurrent_update gdn_recurrent_update_bplus \
            gdn_recurrent_update_bplus_state gdn_recurrent_update_nofma \
            gdn_recurrent_update_bplus_state_multi gdn_verify_norm_pass \
            gdn_recurrent_update_bplus_prenorm gdn_causal_conv1d_update; do
    nm -D --defined-only "$OUT" | grep -q " $name\$" || {
        echo "build_gdn: $OUT is missing $name (the HIP compile failed silently)" >&2
        exit 1
    }
done

echo "built $OUT"
