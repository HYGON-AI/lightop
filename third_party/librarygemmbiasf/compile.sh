#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

rocm_path="${ROCM_PATH:?ROCM_PATH must be set}"

(
    cd "${rocm_path}/.hip" || exit 1
    ./remove_origin.sh
)
hipcc -o test-hytsf test-hytsf.cpp -lhytsf -lsqueue -L.
