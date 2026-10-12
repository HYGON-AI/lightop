#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


(
    cd "${ROCM_PATH}/.hip" || exit 1
    ./remove_origin.sh
)
hipcc -o test-hytsf test-hytsf.cpp -lhytsf -lsqueue -L.
