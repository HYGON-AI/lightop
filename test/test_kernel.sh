
# --------------------------- moe_w8a8 ---------------------- #

HIP_VISIBLE_DEVICES=3 python moe_w8a8/test_kernel_moe_benchmark.py --E 256 --n1-list '[256]'  --n2-list '[7168]' --k-list '[128]' --use-channel-wise
# 生成moe_w8a8_channel-wsie_output.xlsx

# --------------------------- moe_w8a8 ---------------------- #


# --------------------------- moe_marling---------------------- #

HIP_VISIBLE_DEVICES=3 python moe_marlin/moe_w4a16/benchmark_deepseek_lightop.py
#生成"./marlin_triton.xlsx"

# --------------------------- moe_marling ---------------------- #

# --------------------------- w8a8_gemm---------------------- #

HIP_VISIBLE_DEVICES=3 python w8a8/w8a8_test.py
#生成"dpsk_671b.xlsx" qwen3_8b_tp1.xlsx  llama3.3_70b_tp2.xlsx

# --------------------------- w8a8_gemm ---------------------- #