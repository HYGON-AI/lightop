#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


cd $ROCM_PATH/.hip
./remove_origin.sh
cd -
hipcc -o test-hytsf test-hytsf.cpp -lhytsf -lsqueue -L.
