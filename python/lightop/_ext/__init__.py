# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Internal namespace for LightOp's split native extension modules.

Each focused extension (e.g. ``lightop._ext.op_norm``) is installed as a
``.so`` inside this package. See ``lightop/op.py`` for the aggregator that fans
``op.<name>`` out across these modules.
"""
