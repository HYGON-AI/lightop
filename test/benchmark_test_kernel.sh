#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


# --------------------------------------- moe_w8a8 --------------------------------------- #

python moe_w8a8/release_moe_w8a8.py
# 生成moe_w8a8_gemm_channel-wise.xlsx

# -------------------------------------- moe_w8a8 ----------------------------------------------- #


# ----------------------------------- moe_marling----------------------------------------------- #

python moe_marlin/moe_w4a16/benchmark_deepseek_lightop.py
#生成"./marlin_triton.xlsx"

# ----------------------------------- moe_marling----------------------------------------------- #

# ----------------------------------- w8a8_gemm----------------------------------------------- #
python w8a8/w8a8_test.py
#生成"dpsk_671b.xlsx" qwen3_8b_tp1.xlsx  llama3.3_70b_tp2.xlsx

# ----------------------------------- w8a8_gemm----------------------------------------------- #

# ----------------------------------- moe_w4a8_marlin_tp8 ----------------------------------------- #
pytest  moe_marlin/test/tp8/TP_test_deepseek.py
#  pytest的输出格式 如果想测试 其他TP数，修改文件中 "tp_size" 值

# ----------------------------------- moe_w4a8_marlin_ep8 ----------------------------------------- #
pytest  moe_marlin/test/ep8/EP_test_deepseek.py
#  pytest的输出格式 如果想测试 其他EP数，修改文件中 "ep_size" 值

# ----------------------------------- moe_w4a16_marlin_tp_fp16 ------------------------------------# 
python  moe_marlin/test/w4a16/test_deepseek_lightop.py
# 输出 test_w4a16_fused_moe_tp.xlsx 文件
