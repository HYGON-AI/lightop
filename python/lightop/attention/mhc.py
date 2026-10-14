# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 mHC attention APIs."""

from .. import op as lightop_native

mhc_repeat_cp_sglang = lightop_native.mhc_repeat_cp_sglang
supports_mhc_repeat_cp_sglang = lightop_native.supports_mhc_repeat_cp_sglang

__all__ = [
    "mhc_repeat_cp_sglang",
    "supports_mhc_repeat_cp_sglang",
]
