# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 mHC attention APIs."""

try:
    from .. import _mhc_repeat_cp as lightop_native
except ImportError:
    from .. import op as lightop_native

mhc_repeat_cp_sglang = lightop_native.mhc_repeat_cp_sglang
supports_mhc_repeat_cp_sglang = lightop_native.supports_mhc_repeat_cp_sglang

# The existing public API stays usable with older full-library wheels.
if hasattr(lightop_native, "mhc_repeat_cp_sglang_out"):
    mhc_repeat_cp_sglang_out = lightop_native.mhc_repeat_cp_sglang_out

__all__ = [
    "mhc_repeat_cp_sglang",
    "supports_mhc_repeat_cp_sglang",
]
if hasattr(lightop_native, "mhc_repeat_cp_sglang_out"):
    __all__.append("mhc_repeat_cp_sglang_out")
