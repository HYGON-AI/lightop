import torch
from lightop import op
import time
from collections import defaultdict
from typing import Optional
from lightop._lmslim_native.quantize import quant_ops

torch.set_printoptions(precision=4,threshold=100000,linewidth=140,sci_mode=False)
torch.random.manual_seed(0)
repeat_num=100

# 配置选项：是否生成最优配置函数
GENERATE_OPTIMAL_FUNCTION = True  # 设置为True来生成最优配置函数

# 用于存储每个shape的所有配置结果
results_dict = defaultdict(list)

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def _rocblas_scaled_mm(a: torch.Tensor,   #input
                            b: torch.Tensor,  #weight
                            scale_a: torch.Tensor,
                            scale_b: torch.Tensor,
                            out_dtype: torch.dtype,
                            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return quant_ops.rocblas_scaled_mm_nn(a,b,scale_a,scale_b,out_dtype,bias)

def check_config_available(m, n, k, warps, nperblock, kperwarp, splitk, use_n_warps):
    cond1 = m <= 16;
    cond2 = k % kperwarp == 0;
    # cond3 = warps * splitk * kperwarp <= k;
    # cond4 = k % (warps * splitk * kperwarp) == 0;
    if use_n_warps:
        lds_size = warps*nperblock*m*4;
    else:
        lds_size= warps/2*nperblock*m*4 if warps > 1 else nperblock*m*4;
    cond5 = lds_size <= 64*1024*1024;
    part_k = ((k//kperwarp-1)//splitk+1)*kperwarp;
    cond6 = part_k*(splitk-1) < k;
    return cond1  and cond2 and cond5 and cond6;  


def test(m, n, k, warps, nperblock, kperwarp, splitk, use_n_warps=False):
    device = "cuda"
    out_dtype = torch.float16  # 使用fp16
    use_bias = False  # 不使用bias
    
    # 生成测试数据
    a = to_int8(torch.randn((m, k), device=device) * 5)
    b = to_int8(torch.randn((n, k), device=device).t() * 5)
    scale_a = torch.randn((m, 1), device=device, dtype=torch.float32)
    scale_b = torch.randn((n, 1), device=device, dtype=torch.float32)
    bias = None
    
    try:
        # 使用rocblas作为参考实现
        if not check_config_available(m, n, k, warps, nperblock, kperwarp, splitk, use_n_warps):
            print(f"warps={warps}, nperblock={nperblock}, kperwarp={kperwarp}, splitk={splitk}, use_n_warps={use_n_warps} 配置不合适")
            return None
        
        out_ref = _rocblas_scaled_mm(a, b.contiguous().reshape(k,-1), scale_a, scale_b, out_dtype, bias)
        out = op.gemm_w8_a8_smooth_colmajor(a, b, scale_a, scale_b, bias, out_dtype, warps, nperblock, kperwarp, splitk, use_n_warps)
        
        # 验证计算正确性
        is_correct = torch.allclose(out, out_ref, rtol=1e-02, atol=5e-02)
        if not is_correct:
            diff_max = torch.max(torch.abs(out-out_ref))
            print(f"warps={warps}, nperblock={nperblock}, kperwarp={kperwarp}, splitk={splitk}, use_n_warps={use_n_warps} 计算结果不正确, diff_max={diff_max}")
            return None  # 如果计算结果不正确，返回None
        
        # 测试性能
        for i in range(5):  # warm up
            out = op.gemm_w8_a8_smooth_colmajor(a, b, scale_a, scale_b, bias, out_dtype, warps, nperblock, kperwarp, splitk, use_n_warps)
        torch.cuda.synchronize()
        
        start_time = time.time()
        for i in range(repeat_num):
            out = op.gemm_w8_a8_smooth_colmajor(a, b, scale_a, scale_b, bias, out_dtype, warps, nperblock, kperwarp, splitk, use_n_warps)
        torch.cuda.synchronize()
        end_time = time.time()
        
        time_ms = (end_time - start_time) * 1000 / repeat_num
        return time_ms
        
    except Exception as e:
        print(f"配置 warps={warps}, nperblock={nperblock}, kperwarp={kperwarp}, splitk={splitk}, use_n_warps={use_n_warps} 出错: {e}")
        return None

import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1) 

m = [1,2]
nk = [(1536,7168),]
# m = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# nk = list(set([
#     (3584, 5120), (5120, 2560),(5120,13824),(27684,5120),(10240, 5120),(51200,5120),(5120,25600),(5120,2048),(2048,4096),(5120,5120),(5120,4096),(25600,5120),(5120,12800),(2560,2048),(2048,2048),
#     (2560,5120),(5120,2048),(12800,5120),(5120,6400),(1280,2048),(2048,1024),
#     (256,7168),(576,7168),(1536,1536),(2048,512),(2304,7168),(7168,128),(7168,1024),(7168,1152),
#     (1280,4096),(4096,1024),
#     (768,4096),
#     (6144,4096),(4096,4096),(24576,4096),(4096,12288),
#     (28672,8192),(8192,14336),(5120,8192),(8192,4096),
#     (14336,8192),(8192,7168),(2560,8192),(8192,2048),
#     (7168,8192),(8192,3584),(1280,8192),(8192,1024),
#     (512, 7168),(3072, 1536), (1536, 7168), (4096, 512), (4608, 7168), (7168, 256), (7168, 2048), (7168, 2304)
# ]))
warps = [1, 2, 4, 8, 16]
nperblock = [16, 32, 64,]
splitk = [1,2,4,8,10]
kperwarp = [64, 128, 256]
use_n_warps_list = [True, False]

# 测试所有配置
for N, K in nk:
    for M in m:
        shape_key = f"M={M},N={N},K={K}"
        print(f"\n开始测试 {shape_key} 的所有配置...")
        current_configs = []
        
        for use_n_warps in use_n_warps_list:
            for w in warps:
                for nper in nperblock:
                    for kper in kperwarp:
                        # 当use_n_warps=True时，splitk始终为1
                        splitk_values = [1] if use_n_warps else splitk
                        for s in splitk_values:
                            time_ms = test(M, N, K, w, nper, kper, s, use_n_warps)
                            if time_ms is not None:  # 只记录计算正确的结果
                                current_configs.append({
                                    'warps': w,
                                    'nperblock': nper,
                                    'kperwarp': kper,
                                    'splitk': s,
                                    'use_n_warps': use_n_warps,
                                    'time_ms': time_ms
                                })
                                print(f"配置 warps={w}, nperblock={nper}, kperwarp={kper}, splitk={s}, use_n_warps={use_n_warps} 耗时: {time_ms:.3f} ms")
        
        # 每个shape测试完后立即打印最佳配置
        if current_configs:
            best_config = min(current_configs, key=lambda x: x['time_ms'])
            print(f"\n{shape_key} 的最佳配置:")
            print(f"warps={best_config['warps']}, nperblock={best_config['nperblock']}, "
                  f"kperwarp={best_config['kperwarp']}, splitk={best_config['splitk']}, use_n_warps={best_config['use_n_warps']}")
            print(f"最短执行时间: {best_config['time_ms']:.3f} ms")
            print("-" * 80)
            results_dict[shape_key] = current_configs
        else:
            print(f"\n{shape_key}: 未找到有效配置")
            print("-" * 80)

# 最后输出所有shape的最优配置汇总
print("\n\n所有shape的最优配置汇总：")
print("=" * 80)
for shape, configs in results_dict.items():
    if configs:
        best_config = min(configs, key=lambda x: x['time_ms'])
        print(f"{shape}: 最佳配置: warps={best_config['warps']}, nperblock={best_config['nperblock']}, kperwarp={best_config['kperwarp']}, splitk={best_config['splitk']}, use_n_warps={best_config['use_n_warps']}, 执行时间: {best_config['time_ms']:.3f} ms")
    else:
        print(f"\n{shape}: 未找到有效配置")

def generate_optimal_config_function():
    """
    根据tuning结果生成w8a8_smooth_quant_config_dict字典
    """
    # 整理所有shape的最佳配置
    w8a8_smooth_quant_config_dict = {}
    
    for shape_key, configs in results_dict.items():
        if configs:
            best_config = min(configs, key=lambda x: x['time_ms'])
            # 从shape_key中提取M, N, K
            # shape_key格式: "M=1,N=5120,K=13824"
            parts = shape_key.split(',')
            M = int(parts[0].split('=')[1])
            N = int(parts[1].split('=')[1])
            K = int(parts[2].split('=')[1])
            
            # 使用字符串键格式: "M1N5120K13824"
            string_key = f"M{M}N{N}K{K}"
            # 使用元组格式: (warps, nperblock, kperwarp, splitk, use_n_warps)
            w8a8_smooth_quant_config_dict[string_key] = (
                best_config['warps'],
                best_config['nperblock'],
                best_config['kperwarp'],
                best_config['splitk'],
                best_config['use_n_warps']
            )
    
    return w8a8_smooth_quant_config_dict

# 生成并打印最优配置函数
if GENERATE_OPTIMAL_FUNCTION:
    print("\n" + "=" * 80)
    print("生成的w8a8_smooth_quant_config_dict字典：")
    print("=" * 80)
    w8a8_smooth_quant_config_dict = generate_optimal_config_function()
    print("w8a8_smooth_quant_config_dict = {")
    for key, config in w8a8_smooth_quant_config_dict.items():
        print(f"    \"{key}\": {config},")
    print("}")

    # 可选：将字典保存到文件
    import os
    os.makedirs('generator', exist_ok=True)
    with open('generator/w8a8_smooth_quant_config_dict.py', 'w') as f:
        f.write("w8a8_smooth_quant_config_dict = {\n")
        for key, config in w8a8_smooth_quant_config_dict.items():
            f.write(f"    \"{key}\": {config},\n")
        f.write("}\n")

    print("\n" + "=" * 80)
    print("w8a8_smooth_quant_config_dict字典已保存到 generator/w8a8_smooth_quant_config_dict.py")
    print("=" * 80)
