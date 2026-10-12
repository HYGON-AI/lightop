# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native expert-parallel data movement APIs."""

from .. import op as lightop_native

ep_build_m_indices = lightop_native.ep_build_m_indices
ep_gather = lightop_native.ep_gather
ep_scatter = lightop_native.ep_scatter

__all__ = [
    "ep_build_m_indices",
    "ep_gather",
    "ep_scatter",
]
