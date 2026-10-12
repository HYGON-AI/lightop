# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Direct bindings for existing LightOp attention softmax APIs."""

from ..fusesoftmax import FuseLogSoftmax, FuseSoftmax
from ..scaled_masked_softmax import (
    FusedScaleMaskSoftmax,
    FusedScaleMaskSoftmaxDropout,
)

__all__ = [
    "FusedScaleMaskSoftmax",
    "FusedScaleMaskSoftmaxDropout",
    "FuseLogSoftmax",
    "FuseSoftmax",
]
