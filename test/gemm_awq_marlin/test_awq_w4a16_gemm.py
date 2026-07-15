
import torch
import time
from lmslim import quant_ops
from lmslim import quant_ops as ops
from typing import Optional, Type
import os
import pandas as pd
import json
import torch.nn.functional as F
from lmslim.quantize.utils.awq_gemm_get_best_template import AWQ_W4A16_config_cuda
import lmslimquant
import re
import pdb
import lightop
from lightop import gemm_ops
import triton
from datetime import datetime
torch.manual_seed(42)



def data_gen(M,N,K,G,use_random, dtype, use_marlin=False):
    """
    生成 ops.awq_gemm 和 F.linear 所需的 tensor 数据
    返回包含所有必要张量的字典
    """
    device = "cuda"
    group_size = G

    # 检查 group_size 是否能整除 K
    if K % group_size != 0:
        # 尝试其他 group_size
        for gs in [128, 64, 32]:
            if K % gs == 0:
                group_size = gs
                break
        else:
            raise ValueError(f"K={K} 不能被任何标准 group_size (128, 64, 32) 整除")

    # 生成输入 tensor a (M, K) 和原始权重 weight_NN (K, N)
    if use_random:
        a = torch.randn((M, K), device=device, dtype=dtype) / 10
        weight_NN = torch.randn((K, N), device=device, dtype=dtype)
    else:
        a = torch.ones((M, K), device=device, dtype=dtype) / 10
        weight_NN = torch.ones((K, N), device=device, dtype=dtype)

    # 量化权重
    qweight_base, scales, qzeros = awq_quantize_weight(weight_NN, group_size)

    # 转换权重格式
    weight_trans, _sz = ops.convert_s4(qweight_base, qzeros, scales, group_size)

    # 转置权重和 scales_zeros
    weight_trans = weight_trans.reshape(N, -1)
    scales_zeros = ops.sz_permute(_sz)
    scales_zeros = scales_zeros.reshape(N, -1)

    # 处理 padding
    pad_k = K
    padding_group = 0

    # 反量化权重用于 F.linear 基准测试
    dqweight_out = torch.empty((N, K), dtype=dtype).cuda()
    lmslimquant.awq_dequant_colmajor(dqweight_out,
                                     weight_trans,
                                     scales_zeros,
                                     K,
                                     pad_k,  # StrideB_padded
                                     N,
                                     group_size,
                                     N * K * 2)

    # 创建 AWQ workspace
    awqsingleton = AWQShareWorkSpace()
    awqworkspace = awqsingleton.awqworkshapce
    awqworkspacesize = awqsingleton.awqworkshapcesize
    ret = torch.empty((M, N), device=a.device, dtype=a.dtype)
    work_space = torch.empty((110*1024*1024), device=a.device, dtype=torch.float32)
    weight_marlin = gemm_ops.awq_gemm_marlin_weight_repack(weight_trans, N, K)



    # 返回包含所有张量的字典
    data_dict = {
        # 输出
        'ret': ret,
        # 输入
        'a': a,  # (M, K) - 输入矩阵
        'weight_NN': weight_NN,  # (K, N) - 原始权重
        'weight_marlin': weight_marlin,  # (K, N) - marlin权重
        # 量化权重相关
        'weight_trans': weight_trans,  # (N, K//8) - 转换后的量化权重
        'scales_zeros': scales_zeros,  # (N, K//group_size) - 合并的 scales 和 zeros
        'qweight_base': qweight_base,  # (K, N//8) - 基础量化权重
        'scales': scales,  # (K//group_size, N) - 缩放因子
        'qzeros': qzeros,  # (K//group_size, N//8) - 量化零点

        # 反量化权重
        'dqweight_out': dqweight_out,  # (N, K) - 反量化权重，用于 F.linear

        # AWQ workspace
        'awqworkspace': awqworkspace,
        'awqworkspacesize': awqworkspacesize,

        # 参数
        'M': M,
        'N': N,
        'K': K,
        'group_size': group_size,
        'padding_group': padding_group,
        'pad_k': pad_k
    }

    return data_dict

# 复制必要的辅助函数和类
class AWQShareWorkSpace:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AWQShareWorkSpace, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.awqworkshapcesize = self.GetAWQShareWorkspaceSize()
        self.awqworkshapce = self.GetAWQShareWorkspace()

    def GetAWQShareWorkspaceSize(self)->int:
        return quant_ops.GetAWQShareWorkspaceSize()

    def GetAWQShareWorkspace(self)->torch.Tensor:
        return quant_ops.GetAWQShareWorkspace()

def to_int32(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=0, max=1277777)).to(dtype=torch.int32)

def pack_4bit_tensor(tensor):
    # tensor: [group_size, n] or [n], int4 in [0, 15]
    # return: [group_size, n//8] or [1, n//8], int32
    if tensor.dim() == 1:
        # 如果是1D tensor，扩展为2D
        tensor = tensor.unsqueeze(0)  # [1, n]

    k, n = tensor.shape
    assert n % 8 == 0
    tensor = tensor.clone()
    tensor = tensor & 0xF  # 只保留低4位
    tensor = tensor.view(k, n // 8, 8)  # [k, n//8, 8]
    packed = torch.zeros((k, n // 8), dtype=torch.int32, device=tensor.device)
    for i in range(8):
        packed |= (tensor[:, :, i].to(torch.int32) << (i * 4))
    return packed

def awq_quantize_weight(weight, group_size=128):
    k, n = weight.shape
    assert k % group_size == 0
    qweight = torch.zeros((k, n // 8), dtype=torch.int32, device=weight.device)
    scales = torch.zeros((k // group_size, n), dtype=torch.float16, device=weight.device)
    qzeros = torch.zeros((k // group_size, n // 8), dtype=torch.int32, device=weight.device)

    for g in range(k // group_size):
        w_group = weight[g * group_size:(g + 1) * group_size, :]  # [group_size, n]

        # 1. 计算每列的最小值和最大值
        w_min = w_group.min(dim=0)[0]  # [n]
        w_max = w_group.max(dim=0)[0]  # [n]

        # 2. 计算 scale 和 zero-point (非对称量化)
        scale = (w_max - w_min) / 15.0  # 4bit范围是0-15
        scale[scale == 0] = 1.0  # 防止除零
        zero_point = w_min / scale  # 计算zero-point

        scales[g, :] = scale

        # 3. 量化: (weight - zero_point) / scale
        q = torch.round(w_group / scale - zero_point).clamp(0, 15).to(torch.int8)

        # 4. pack qweight
        q_packed = pack_4bit_tensor(q)
        qweight[g * group_size:(g + 1) * group_size, :] = q_packed

        # 5. pack qzeros
        qzeros_packed = pack_4bit_tensor(zero_point.to(torch.int8))
        qzeros[g, :] = qzeros_packed.squeeze(0)  # 去掉batch维度

    return qweight, scales, qzeros



def tensor_vs(tensor1: torch.tensor,tensor2: torch.tensor, M,N,K,G, use_random, dtype, mode="test", use_marlin=False):
    preceis_ori = torch.allclose(tensor1, tensor2, rtol=1e-03, atol=1e-2)
    if mode == "no_print":
        return preceis_ori
    if preceis_ori:
        print(f" 验证通过: 输出结果与预期一致, M={M}, N={N}, K={K}, G={G}, use_random={use_random}, dtype={dtype}, use_marlin={use_marlin}")
        return True
  
    if mode == "test":
        absolute_errors = torch.abs(tensor1 - tensor2)
        max_error = torch.max(absolute_errors).item()
        mean_error = torch.mean(absolute_errors).item()
        print(f"{M}x{N}x{K}x{G} Absolute Errors: {absolute_errors}; Max Error: {max_error}; Mean Error: {mean_error}; use_random={use_random}, dtype={dtype}, use_marlin={use_marlin}")
    return False








def check_config_available(m, n, k, num_k_warps, num_n_warps, k_per_warp, n_per_warp, splitk):
    # cond1 = m <= 16
    cond1 = True
    m = min(m, 16)
    cond2 = k % k_per_warp == 0

    if splitk > 1:
        if num_k_warps > 1:
            lds_size = num_k_warps/2*n_per_warp*m*4 # 4表示float占4字节
        else:
            lds_size = 0
    else:
        if num_k_warps > 1:
            lds_size = num_k_warps/2*n_per_warp*m*4 # 4表示float占4字节
        else:
            lds_size = n_per_warp*m*2 # 2表示half_t占2字节

    all_lds_size = lds_size * num_n_warps
    cond3 = all_lds_size <= 64*1024*1024
    part_k = ((k//k_per_warp-1)//splitk+1)*k_per_warp
    cond4 = part_k*(splitk-1) < k
    cond5 = num_k_warps*num_n_warps <=16
    cond6 = k_per_warp <= k
    cond7 = n_per_warp <= n
    cond8 = True
    cond9 = k_per_warp == 256
    if k%256 != 0 and k%128 == 0 and k_per_warp == 128:
        cond9 = True
    # pdb.set_trace()
    return  cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond9


def sing_test_dequant(N,K, G, dtype, test_speed, use_random, use_marlin):
    assert use_marlin == True, "use_marlin must be True"
    data_dict = data_gen(16,N,K,G,use_random=use_random, dtype=dtype, use_marlin=use_marlin)
    ref = data_dict['dqweight_out']
    ret = torch.empty((N,K), device=ref.device, dtype=ref.dtype)
    weight_quant = data_dict['weight_trans'] if not use_marlin else data_dict['weight_marlin']
    gemm_ops.gemm_awq_w4a16_marlin_dequant(ret, weight_quant, data_dict['scales_zeros'], use_marlin)
    preceis_ori = torch.allclose(ref, ret, rtol=1e-03, atol=1e-2)
    # pdb.set_trace()
    if preceis_ori:
        print(f" 验证通过: 输出结果与预期一致, N={N}, K={K}, G={G}, dtype={dtype}, use_random={use_random}, use_marlin={use_marlin}")
    else:
        print(f"N={N}, K={K}, G={G}, dtype={dtype}, use_random={use_random}, use_marlin={use_marlin} 不正确")
    if test_speed:
        for i in range(10):
            gemm_ops.gemm_awq_w4a16_marlin_dequant(ret, weight_quant, data_dict['scales_zeros'], use_marlin)
        torch.cuda.synchronize()
        marlin_dequant_time = triton.testing.do_bench(lambda:gemm_ops.gemm_awq_w4a16_marlin_dequant(ret, weight_quant, data_dict['scales_zeros'], use_marlin),quantiles=None, return_mode="mean")*1000
        marlin_data_bytes = ret.element_size() * ret.nelement() + weight_quant.element_size() * weight_quant.nelement() + data_dict['scales_zeros'].element_size() * data_dict['scales_zeros'].nelement()
        data_bytes_gb = marlin_data_bytes / (2**30)
        marlin_bandwidth = data_bytes_gb / (marlin_dequant_time/1000/1000)
        no_marlin_time = triton.testing.do_bench(lambda: lmslimquant.awq_dequant_colmajor(data_dict['dqweight_out'],
                                     data_dict['weight_trans'],
                                     data_dict['scales_zeros'],
                                     data_dict['K'],
                                     data_dict['K'],  # StrideB_padded
                                     data_dict['N'],
                                     data_dict['group_size'],
                                     data_dict['N'] * data_dict['K'] * 2),quantiles=None, return_mode="mean")*1000
        no_marlin_bandwidth = data_bytes_gb / (no_marlin_time/1000/1000)
        print(f"awq_w4a16_dequant算子速度测试结果 - N={N}, K={K}, G={G}, marlin_dequant_time: {marlin_dequant_time:.2f} us, no_marlin_dequant_time: {no_marlin_time:.2f} us, marlin_bandwidth: {marlin_bandwidth:.2f} GB/s, no_marlin_bandwidth: {no_marlin_bandwidth:.2f} GB/s")


def multi_test_dequant(mode_name, G, dtype, test_speed, use_random, use_marlin):
    N_list = model_shape_dict[mode_name]["N"]
    K_list = model_shape_dict[mode_name]["K"]
    for i in range(len(N_list)):
        N = N_list[i]
        K = K_list[i]
        sing_test_dequant(N=N,K=K, G=G, dtype=dtype, test_speed=test_speed, use_random=use_random, use_marlin=use_marlin)


def single_test(M,N,K, G, test_speed, use_random, dtype, use_marlin):
    # print(f"single_test: M={M}, N={N}, K={K}, G={G}, test_speed={test_speed}, use_random={use_random}")
    data_dict = data_gen(M,N,K,G,use_random, dtype, use_marlin=use_marlin)
    if use_marlin:
        weight_quant = data_dict['weight_marlin']
    for i in range(10):
        out_de_fp16 = F.linear(data_dict['a'],data_dict['dqweight_out'])
        out_awq = ops.awq_gemm(data_dict['a'],data_dict['weight_trans'],data_dict['scales_zeros'],data_dict['M'],data_dict['N'],data_dict['K'],data_dict['group_size'],0,
                                    data_dict['awqworkspace'],
                                    data_dict['awqworkspacesize'])
        out_lightop = gemm_ops.gemm_awq_w4a16_marlin(data_dict['a'], weight_quant, data_dict['scales_zeros'])
    torch.cuda.synchronize()
    # out_lightop = data_dict['ret']
    # out_lightop = lightop.gemm_awq_w4a16(data_dict['a'],data_dict['weight_trans'],data_dict['scales_zeros'])
    preceis_ori = tensor_vs(out_de_fp16, out_awq, M,N,K,G, use_random, dtype, use_marlin=use_marlin, mode="no_print") and tensor_vs(out_de_fp16, out_lightop, M,N,K,G, use_random, dtype, use_marlin=use_marlin, mode="no_print")
    # pdb.set_trace()
    if not preceis_ori:
        print(f"M{M}xN{N}xK{K} 不正确")
        return None

    result = {
        'M': M,
        'N': N,
        'K': K,
        'G': G,
        'correct': True
    }

    if test_speed:
        awq_elapsed_time = triton.testing.do_bench(lambda:ops.awq_gemm(data_dict['a'],data_dict['weight_trans'],data_dict['scales_zeros'],data_dict['M'],data_dict['N'],data_dict['K'],data_dict['group_size'],0,
                                data_dict['awqworkspace'],
                                data_dict['awqworkspacesize']),quantiles=None, return_mode="mean")*1000
        awq_data_bytes = data_dict['a'].element_size() * data_dict['a'].nelement() + data_dict['weight_trans'].element_size() * data_dict['weight_trans'].nelement() + data_dict['scales_zeros'].element_size() * data_dict['scales_zeros'].nelement() + out_awq.element_size() * out_awq.nelement()
        awq_data_bytes_gb = awq_data_bytes / (2**30)
        awq_bandwidth = awq_data_bytes_gb / (awq_elapsed_time/1000/1000)

        fp16_elapsed_time = triton.testing.do_bench(lambda:F.linear(data_dict['a'],data_dict['dqweight_out']),quantiles=None, return_mode="mean")*1000
        fp16_data_bytes = data_dict['a'].element_size() * data_dict['a'].nelement() + data_dict['dqweight_out'].element_size() * data_dict['dqweight_out'].nelement() + out_de_fp16.element_size() * out_de_fp16.nelement()
        fp16_data_bytes_gb = fp16_data_bytes / (2**30)
        fp16_bandwidth = fp16_data_bytes_gb / (fp16_elapsed_time/1000/1000)
        preceis_ori = tensor_vs(out_de_fp16, out_awq, M,N,K,G, use_random, dtype, use_marlin=use_marlin, mode="no_print") and tensor_vs(out_de_fp16, out_lightop, M,N,K,G, use_random, dtype, use_marlin=use_marlin, mode="no_print")
        lightop_elapsed_time = triton.testing.do_bench(lambda:gemm_ops.gemm_awq_w4a16_marlin(data_dict['a'], weight_quant, data_dict['scales_zeros']),quantiles=None, return_mode="mean")*1000
        lightop_bandwidth = awq_data_bytes_gb / (lightop_elapsed_time/1000/1000)

        result.update({
            'lightop_elapsed_time': lightop_elapsed_time,
            'awq_elapsed_time': awq_elapsed_time,
            'fp16_elapsed_time': fp16_elapsed_time,
            'lightop_bandwidth': lightop_bandwidth,
            'awq_bandwidth': awq_bandwidth,
            'fp16_bandwidth': fp16_bandwidth
        })

        print(f"M{M}xN{N}xK{K}正确, marlin_time={lightop_elapsed_time:.2f} us, awq_old_time={awq_elapsed_time:.2f} us, fp16_time={fp16_elapsed_time:.2f} us, marlin_bandwidth={lightop_bandwidth:.2f} GB/s, awq_old_bandwidth={awq_bandwidth:.2f} GB/s, fp16_bandwidth={fp16_bandwidth:.2f} GB/s")

    return result



def single_test_with_config(M,N,K,G, num_n_warps, num_k_warps, n_per_warp, k_per_warp, splitk, dtype, data_dict=None, use_random=True, test_speed=True, use_marlin=False, mode="tunning"):
    if data_dict is None:
        data_dict = data_gen(M,N,K,G,use_random=use_random, dtype=dtype, use_marlin=use_marlin)
    out_de_fp16 = F.linear(data_dict['a'],data_dict['dqweight_out'])
    weight_quant = data_dict['weight_trans'] if not use_marlin else data_dict['weight_marlin']
    data_dict['ret'] = torch.zeros((M,N), device=data_dict['a'].device, dtype=data_dict['a'].dtype)
    for i in range(10):
        lightop.op.gemm_awq_w4a16(data_dict['ret'], data_dict['a'], weight_quant, data_dict['scales_zeros'], num_n_warps, num_k_warps, n_per_warp, k_per_warp, splitk, use_marlin)
    Correctness = tensor_vs(out_de_fp16, data_dict['ret'], M,N,K,G, use_random = True, dtype=dtype, mode=mode, use_marlin=use_marlin)
    # pdb.set_trace()
    if test_speed == False:
        return 999999.0
    if Correctness:
        # warmup
        for i in range(10):
            F.linear(data_dict['a'], data_dict['dqweight_out'])
        torch.cuda.synchronize()
        elapsed_time = triton.testing.do_bench(lambda:lightop.op.gemm_awq_w4a16(data_dict['ret'], data_dict['a'], weight_quant, data_dict['scales_zeros'], num_n_warps, num_k_warps, n_per_warp, k_per_warp, splitk, use_marlin),quantiles=None, return_mode="mean")*1000
        Correctness = tensor_vs(out_de_fp16, data_dict['ret'], M,N,K,G, use_random = True, dtype=dtype, mode=mode, use_marlin=use_marlin)
        if not Correctness:
            return 999999.0, 0.0, 0.0
        tflops = M*N*K*2*1e-6 / elapsed_time
        data_bytes = data_dict['ret'].element_size() * data_dict['ret'].nelement() + data_dict['a'].element_size() * data_dict['a'].nelement() + weight_quant.element_size() * weight_quant.nelement() + data_dict['scales_zeros'].element_size() * data_dict['scales_zeros'].nelement()
        data_bytes_gb = data_bytes / (2**30)
        bandwidth = data_bytes_gb / (elapsed_time/1000/1000)
        # pdb.set_trace()
        print(f"M={M},N={N},K={K},G={G} 配置 num_n_warps={num_n_warps}, num_k_warps={num_k_warps}, n_per_warp={n_per_warp}, k_per_warp={k_per_warp}, splitk={splitk} 正确, 速度测试结果: {elapsed_time:.2f} us, 算力: {tflops:.2f} TFlops, Bandwidth: {bandwidth:.2f} GB/s", flush=True)
        return elapsed_time, tflops, bandwidth
    else:
        print(f"M={M},N={N},K={K},G={G} 配置 num_n_warps={num_n_warps}, num_k_warps={num_k_warps}, n_per_warp={n_per_warp}, k_per_warp={k_per_warp}, splitk={splitk} 不正确", flush=True)
        return 999999.0, 0.0, 0.0

def single_tunning(M,N,K, G, dtype, use_marlin):
    num_k_warps_list = [1, 2, 4, 8, 16]
    num_n_warps_list = [1, 2]
    k_per_warp_list = [256,128,]
    n_per_warp_list = [16,32,64,128]
    splitk_list = [1,2,4,8,10]

    # num_k_warps_list = [1, ]
    # num_n_warps_list = [2, ]
    # k_per_warp_list = [128,]
    # n_per_warp_list = [128, ]
    # splitk_list = [1,]
    shape_key = f"M={M},N={N},K={K},G={G}"
    data_dict = data_gen(M,N,K,G,use_random=True, dtype=dtype, use_marlin=use_marlin)
    print(f"\n开始测试 {shape_key} 的所有配置...", flush=True)
    num_available_configs = 0
    best_config = None
    best_time = 999999.0
    best_tflops = 0.0
    best_bandwidth = 0.0
    for num_k_warps in num_k_warps_list:
        for num_n_warps in num_n_warps_list:
            for k_per_warp in k_per_warp_list:
                for n_per_warp in n_per_warp_list:
                    for splitk in splitk_list:
                        if check_config_available(M,N,K, num_k_warps, num_n_warps, k_per_warp, n_per_warp, splitk):
                            print(f"配置 num_n_warps={num_n_warps}, num_k_warps={num_k_warps}, n_per_warp={n_per_warp}, k_per_warp={k_per_warp}, splitk={splitk} 可用", flush=True)
                            num_available_configs += 1
                            time, tflops, bandwidth = single_test_with_config(M,N,K,G, num_n_warps, num_k_warps, n_per_warp, k_per_warp, splitk, dtype, data_dict, use_random=True, use_marlin=use_marlin)
                            if time < best_time:
                                best_time = time
                                best_tflops = tflops
                                best_bandwidth = bandwidth
                                best_config = f"num_n_warps={num_n_warps}, num_k_warps={num_k_warps}, n_per_warp={n_per_warp}, k_per_warp={k_per_warp}, splitk={splitk}, use_marlin={use_marlin}"
    result_str = f"{shape_key} 可用配置数量: {num_available_configs}, 最佳配置: {best_config}, 最佳耗时: {best_time:.2f} us, 最佳算力: {best_tflops:.2f} TFlops, 最佳Bandwidth: {best_bandwidth:.2f} GB/s"
    print(result_str, flush=True)
    return result_str


model_shape_dict = {
    "llama2": {
        "N": [12288, 4096, 22016, 4096, 6144, 4096, 11008, 4096, 3072, 4096, 5504, 15360, 5120, 27648],
        "K": [4096, 4096, 4096, 11008, 4096, 2048, 4096, 5504, 4096, 1024, 4096, 5120, 5120, 5120],
    },
    "dpsk_671b": {
        "N": [256, 576, 1536, 1536, 2048, 2304, 7168, 7168, 7168],
        "K": [7168, 7168, 1536, 7168, 512, 7168, 128, 1024, 1152],
    },
    "qwen25_72b": {
        "N": [10240, 8192, 59392, 8192, 5120, 8192, 29696, 8192, 2560, 8192, 14848, 8192, 1280, 8192, 7424, 8192],
        "K": [8192, 8192, 8192, 29696, 8192, 4096, 8192, 14848, 8192, 2048, 8192, 7424, 8192, 1024, 8192, 3712],
    },
}

def multi_tunning(mode_name, G, dtype, use_marlin):
    # M_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    # M_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,48,64,80,96,112,128,128+64,256]
    # M_list = [x+16 for x in [1,2,4,8,16]] 
    # M_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    M_list = [16, ]
    N_list = model_shape_dict[mode_name]["N"]
    K_list = model_shape_dict[mode_name]["K"]
    best_config_dict = {}
    best_config_list = []
    for M in M_list:
        for i in range(len(N_list)):
            N = N_list[i]
            K = K_list[i]
            best_config_info = single_tunning(M,N,K,G,dtype,use_marlin)
            best_config_list.append(best_config_info)
            import re
            config_match = re.search(r'num_n_warps=(\d+), num_k_warps=(\d+), n_per_warp=(\d+), k_per_warp=(\d+), splitk=(\d+)', best_config_info)
            if config_match:
                num_n_warps = int(config_match.group(1))
                num_k_warps = int(config_match.group(2))
                n_per_warp = int(config_match.group(3))
                k_per_warp = int(config_match.group(4))
                splitk = int(config_match.group(5))

                key = f"M{M}N{N}K{K}"
                best_config_dict[key] = (num_n_warps, num_k_warps, n_per_warp, k_per_warp, splitk)

    for i in range(len(best_config_list)):
        print(best_config_list[i], flush=True)

    gpu_cus = int(os.environ.get("LIGHTOP_GPU_CUS", "80"))
    arch_gfx = str(os.environ.get("LIGHTOP_GPU_TARGET", "gfx936"))

    print("\n最佳配置字典:", flush=True)
    print(f"awq_w4a16_gemm_config_dict_{gpu_cus}_{arch_gfx} = {{", flush=True)
    for key, config in best_config_dict.items():
        print(f'    "{key}": {config},', flush=True)
    print("}", flush=True)

    return best_config_dict



llama2_awq_w4a16_gemm_config_dict_80_gfx936 = {
    "M16N12288K4096": (1, 8, 64, 128, 2),
    "M16N4096K4096": (1, 16, 64, 128, 1),
    "M16N22016K4096": (1, 8, 64, 128, 2),
    "M16N4096K11008": (1, 4, 64, 128, 10),
    "M16N6144K4096": (1, 4, 64, 128, 4),
    "M16N4096K2048": (1, 8, 64, 128, 1),
    "M16N11008K4096": (1, 4, 64, 128, 4),
    "M16N4096K5504": (1, 16, 64, 128, 1),
    "M16N3072K4096": (1, 4, 64, 128, 4),
    "M16N4096K1024": (1, 4, 32, 128, 1),
    "M16N5504K4096": (1, 4, 64, 128, 4),
    "M16N15360K5120": (1, 4, 128, 128, 2),
    "M16N5120K5120": (1, 16, 64, 128, 1),
    "M16N27648K5120": (1, 4, 128, 128, 1),
}

qwen25_72b_awq_w4a16_gemm_config_dict_80_gfx936 = {
    "M1N10240K8192": (1, 16, 64, 256, 1),
    "M1N8192K8192": (1, 4, 64, 256, 8),
    "M1N59392K8192": (1, 16, 64, 256, 1),
    "M1N8192K29696": (1, 8, 64, 256, 8),
    "M1N5120K8192": (1, 16, 64, 256, 1),
    "M1N8192K4096": (1, 2, 64, 256, 8),
    "M1N29696K8192": (1, 16, 64, 256, 1),
    "M1N8192K14848": (1, 4, 64, 256, 10),
    "M1N2560K8192": (1, 8, 32, 256, 1),
    "M1N8192K2048": (1, 2, 32, 256, 4),
    "M1N14848K8192": (1, 16, 64, 256, 1),
    "M1N8192K7424": (2, 2, 64, 256, 10),
    "M1N1280K8192": (1, 8, 16, 256, 1),
    "M1N8192K1024": (1, 2, 32, 256, 1),
    "M1N7424K8192": (1, 8, 64, 256, 2),
    "M1N8192K3712": (1, 2, 32, 128, 4),
    "M2N10240K8192": (1, 16, 128, 256, 1),
    "M2N8192K8192": (1, 4, 64, 256, 8),
    "M2N59392K8192": (1, 16, 64, 256, 1),
    "M2N8192K29696": (1, 8, 64, 256, 8),
    "M2N5120K8192": (1, 16, 64, 256, 1),
    "M2N8192K4096": (1, 2, 64, 256, 8),
    "M2N29696K8192": (1, 16, 64, 256, 1),
    "M2N8192K14848": (1, 4, 64, 256, 10),
    "M2N2560K8192": (1, 8, 32, 256, 1),
    "M2N8192K2048": (1, 2, 32, 256, 4),
    "M2N14848K8192": (1, 16, 64, 256, 1),
    "M2N8192K7424": (2, 2, 64, 256, 10),
    "M2N1280K8192": (1, 8, 16, 256, 1),
    "M2N8192K1024": (2, 2, 32, 256, 1),
    "M2N7424K8192": (1, 8, 64, 256, 2),
    "M2N8192K3712": (2, 2, 32, 128, 4),
    "M3N10240K8192": (1, 16, 128, 256, 1),
    "M3N8192K8192": (1, 4, 64, 256, 8),
    "M3N59392K8192": (1, 16, 64, 256, 1),
    "M3N8192K29696": (1, 8, 64, 256, 8),
    "M3N5120K8192": (1, 16, 64, 256, 1),
    "M3N8192K4096": (1, 2, 64, 256, 8),
    "M3N29696K8192": (1, 16, 64, 256, 1),
    "M3N8192K14848": (1, 4, 64, 256, 10),
    "M3N2560K8192": (1, 16, 32, 256, 1),
    "M3N8192K2048": (1, 2, 32, 256, 4),
    "M3N14848K8192": (1, 16, 64, 256, 1),
    "M3N8192K7424": (2, 2, 64, 256, 10),
    "M3N1280K8192": (1, 4, 32, 256, 8),
    "M3N8192K1024": (2, 2, 32, 256, 1),
    "M3N7424K8192": (1, 4, 64, 256, 4),
    "M3N8192K3712": (1, 2, 32, 128, 4),
    "M4N10240K8192": (1, 16, 128, 256, 1),
    "M4N8192K8192": (1, 4, 64, 256, 8),
    "M4N59392K8192": (1, 16, 64, 256, 1),
    "M4N8192K29696": (1, 8, 128, 256, 10),
    "M4N5120K8192": (1, 16, 64, 256, 1),
    "M4N8192K4096": (1, 2, 64, 256, 8),
    "M4N29696K8192": (1, 16, 64, 256, 1),
    "M4N8192K14848": (1, 4, 64, 256, 10),
    "M4N2560K8192": (1, 2, 64, 256, 8),
    "M4N8192K2048": (1, 8, 64, 256, 1),
    "M4N14848K8192": (1, 16, 64, 256, 1),
    "M4N8192K7424": (2, 2, 64, 256, 10),
    "M4N1280K8192": (1, 4, 64, 256, 8),
    "M4N8192K1024": (1, 1, 16, 256, 1),
    "M4N7424K8192": (1, 4, 64, 256, 4),
    "M4N8192K3712": (1, 2, 64, 128, 8),
    "M5N10240K8192": (1, 16, 128, 256, 1),
    "M5N8192K8192": (1, 4, 64, 256, 8),
    "M5N59392K8192": (1, 16, 64, 256, 1),
    "M5N8192K29696": (1, 8, 64, 256, 8),
    "M5N5120K8192": (1, 16, 64, 256, 1),
    "M5N8192K4096": (1, 2, 64, 256, 8),
    "M5N29696K8192": (1, 16, 64, 256, 1),
    "M5N8192K14848": (1, 4, 64, 256, 10),
    "M5N2560K8192": (1, 2, 64, 256, 8),
    "M5N8192K2048": (1, 8, 64, 256, 1),
    "M5N14848K8192": (1, 16, 64, 256, 1),
    "M5N8192K7424": (2, 2, 64, 256, 10),
    "M5N1280K8192": (1, 4, 32, 256, 8),
    "M5N8192K1024": (1, 4, 64, 256, 1),
    "M5N7424K8192": (1, 8, 64, 256, 2),
    "M5N8192K3712": (1, 2, 32, 128, 4),
    "M6N10240K8192": (1, 16, 128, 256, 1),
    "M6N8192K8192": (1, 4, 64, 256, 8),
    "M6N59392K8192": (1, 16, 64, 256, 1),
    "M6N8192K29696": (1, 8, 128, 256, 10),
    "M6N5120K8192": (1, 16, 64, 256, 1),
    "M6N8192K4096": (1, 2, 64, 256, 8),
    "M6N29696K8192": (1, 8, 128, 256, 1),
    "M6N8192K14848": (2, 4, 128, 256, 8),
    "M6N2560K8192": (1, 4, 64, 256, 8),
    "M6N8192K2048": (2, 8, 64, 256, 1),
    "M6N14848K8192": (1, 16, 64, 256, 1),
    "M6N8192K7424": (2, 2, 64, 256, 10),
    "M6N1280K8192": (1, 4, 32, 256, 8),
    "M6N8192K1024": (1, 4, 64, 256, 1),
    "M6N7424K8192": (1, 4, 128, 256, 8),
    "M6N8192K3712": (1, 2, 32, 128, 4),
    "M7N10240K8192": (1, 16, 128, 256, 1),
    "M7N8192K8192": (1, 2, 64, 256, 8),
    "M7N59392K8192": (1, 8, 128, 256, 1),
    "M7N8192K29696": (1, 8, 128, 256, 10),
    "M7N5120K8192": (1, 16, 64, 256, 1),
    "M7N8192K4096": (1, 2, 64, 256, 8),
    "M7N29696K8192": (1, 8, 128, 256, 1),
    "M7N8192K14848": (2, 4, 128, 256, 8),
    "M7N2560K8192": (1, 4, 64, 256, 8),
    "M7N8192K2048": (1, 8, 128, 256, 1),
    "M7N14848K8192": (1, 16, 64, 256, 1),
    "M7N8192K7424": (1, 2, 64, 256, 8),
    "M7N1280K8192": (1, 4, 32, 256, 8),
    "M7N8192K1024": (1, 4, 64, 256, 1),
    "M7N7424K8192": (1, 4, 128, 256, 8),
    "M7N8192K3712": (1, 4, 64, 128, 4),
    "M8N10240K8192": (1, 16, 128, 256, 1),
    "M8N8192K8192": (1, 4, 128, 256, 8),
    "M8N59392K8192": (1, 8, 128, 256, 1),
    "M8N8192K29696": (1, 8, 128, 256, 10),
    "M8N5120K8192": (1, 16, 64, 256, 1),
    "M8N8192K4096": (1, 4, 64, 256, 4),
    "M8N29696K8192": (1, 8, 128, 256, 1),
    "M8N8192K14848": (2, 4, 128, 256, 8),
    "M8N2560K8192": (1, 4, 64, 256, 8),
    "M8N8192K2048": (1, 8, 128, 256, 1),
    "M8N14848K8192": (1, 8, 128, 256, 2),
    "M8N8192K7424": (1, 2, 64, 256, 8),
    "M8N1280K8192": (1, 4, 32, 256, 8),
    "M8N8192K1024": (1, 4, 64, 256, 1),
    "M8N7424K8192": (1, 8, 128, 256, 4),
    "M8N8192K3712": (1, 4, 64, 128, 4),
    "M9N10240K8192": (1, 16, 128, 256, 1),
    "M9N8192K8192": (1, 4, 128, 256, 8),
    "M9N59392K8192": (1, 8, 128, 256, 1),
    "M9N8192K29696": (1, 8, 128, 256, 10),
    "M9N5120K8192": (1, 16, 64, 256, 1),
    "M9N8192K4096": (1, 16, 128, 256, 1),
    "M9N29696K8192": (1, 8, 128, 256, 1),
    "M9N8192K14848": (2, 4, 128, 256, 8),
    "M9N2560K8192": (1, 4, 64, 256, 8),
    "M9N8192K2048": (1, 8, 128, 256, 1),
    "M9N14848K8192": (1, 8, 128, 256, 2),
    "M9N8192K7424": (1, 2, 64, 256, 8),
    "M9N1280K8192": (1, 4, 32, 256, 8),
    "M9N8192K1024": (1, 4, 64, 256, 1),
    "M9N7424K8192": (1, 8, 128, 256, 4),
    "M9N8192K3712": (1, 4, 64, 128, 4),
    "M10N10240K8192": (1, 16, 128, 256, 1),
    "M10N8192K8192": (1, 4, 128, 256, 8),
    "M10N59392K8192": (1, 8, 128, 256, 1),
    "M10N8192K29696": (1, 8, 128, 256, 10),
    "M10N5120K8192": (1, 16, 64, 256, 1),
    "M10N8192K4096": (1, 16, 128, 256, 1),
    "M10N29696K8192": (1, 8, 128, 256, 1),
    "M10N8192K14848": (2, 4, 128, 256, 8),
    "M10N2560K8192": (1, 4, 64, 256, 8),
    "M10N8192K2048": (1, 8, 128, 256, 1),
    "M10N14848K8192": (1, 8, 128, 256, 2),
    "M10N8192K7424": (1, 4, 64, 256, 4),
    "M10N1280K8192": (1, 4, 64, 256, 8),
    "M10N8192K1024": (1, 4, 64, 256, 1),
    "M10N7424K8192": (1, 8, 128, 256, 4),
    "M10N8192K3712": (1, 4, 64, 128, 4),
    "M11N10240K8192": (1, 16, 128, 256, 1),
    "M11N8192K8192": (1, 4, 128, 256, 8),
    "M11N59392K8192": (1, 8, 128, 256, 1),
    "M11N8192K29696": (1, 4, 128, 256, 10),
    "M11N5120K8192": (1, 16, 64, 256, 1),
    "M11N8192K4096": (1, 8, 128, 256, 1),
    "M11N29696K8192": (1, 8, 128, 256, 1),
    "M11N8192K14848": (1, 4, 128, 256, 8),
    "M11N2560K8192": (1, 4, 64, 256, 8),
    "M11N8192K2048": (1, 8, 128, 256, 1),
    "M11N14848K8192": (1, 8, 128, 256, 2),
    "M11N8192K7424": (1, 4, 64, 256, 4),
    "M11N1280K8192": (1, 4, 64, 256, 8),
    "M11N8192K1024": (1, 4, 64, 256, 1),
    "M11N7424K8192": (1, 8, 64, 256, 2),
    "M11N8192K3712": (1, 4, 64, 128, 4),
    "M12N10240K8192": (1, 16, 128, 256, 1),
    "M12N8192K8192": (1, 4, 128, 256, 8),
    "M12N59392K8192": (1, 8, 128, 256, 1),
    "M12N8192K29696": (1, 4, 128, 256, 10),
    "M12N5120K8192": (1, 4, 64, 256, 4),
    "M12N8192K4096": (1, 8, 128, 256, 1),
    "M12N29696K8192": (1, 8, 128, 256, 1),
    "M12N8192K14848": (1, 4, 128, 256, 8),
    "M12N2560K8192": (1, 4, 64, 256, 8),
    "M12N8192K2048": (1, 8, 128, 256, 1),
    "M12N14848K8192": (1, 8, 128, 256, 2),
    "M12N8192K7424": (1, 4, 64, 256, 4),
    "M12N1280K8192": (1, 4, 64, 256, 8),
    "M12N8192K1024": (1, 4, 64, 256, 1),
    "M12N7424K8192": (1, 4, 128, 256, 4),
    "M12N8192K3712": (1, 4, 64, 128, 4),
    "M13N10240K8192": (1, 16, 128, 256, 1),
    "M13N8192K8192": (1, 16, 128, 256, 1),
    "M13N59392K8192": (1, 8, 128, 256, 1),
    "M13N8192K29696": (1, 4, 128, 256, 10),
    "M13N5120K8192": (1, 16, 64, 256, 1),
    "M13N8192K4096": (1, 8, 128, 256, 1),
    "M13N29696K8192": (1, 8, 128, 256, 1),
    "M13N8192K14848": (1, 4, 128, 256, 8),
    "M13N2560K8192": (1, 4, 64, 256, 8),
    "M13N8192K2048": (1, 8, 128, 256, 1),
    "M13N14848K8192": (1, 8, 128, 256, 2),
    "M13N8192K7424": (1, 4, 64, 256, 4),
    "M13N1280K8192": (1, 4, 64, 256, 8),
    "M13N8192K1024": (1, 4, 64, 256, 1),
    "M13N7424K8192": (1, 4, 128, 256, 4),
    "M13N8192K3712": (1, 4, 64, 128, 4),
    "M14N10240K8192": (1, 16, 128, 256, 1),
    "M14N8192K8192": (1, 16, 128, 256, 1),
    "M14N59392K8192": (1, 8, 128, 256, 1),
    "M14N8192K29696": (1, 4, 128, 256, 10),
    "M14N5120K8192": (1, 16, 64, 256, 1),
    "M14N8192K4096": (1, 8, 128, 256, 1),
    "M14N29696K8192": (1, 8, 128, 256, 1),
    "M14N8192K14848": (1, 4, 128, 256, 8),
    "M14N2560K8192": (1, 4, 64, 256, 8),
    "M14N8192K2048": (1, 8, 128, 256, 1),
    "M14N14848K8192": (1, 8, 128, 256, 2),
    "M14N8192K7424": (1, 4, 64, 256, 4),
    "M14N1280K8192": (1, 4, 64, 256, 8),
    "M14N8192K1024": (2, 4, 64, 256, 1),
    "M14N7424K8192": (1, 4, 128, 256, 4),
    "M14N8192K3712": (1, 4, 64, 128, 4),
    "M15N10240K8192": (1, 16, 128, 256, 1),
    "M15N8192K8192": (1, 16, 128, 256, 1),
    "M15N59392K8192": (1, 8, 128, 256, 1),
    "M15N8192K29696": (1, 4, 128, 256, 10),
    "M15N5120K8192": (1, 4, 64, 256, 4),
    "M15N8192K4096": (1, 8, 128, 256, 1),
    "M15N29696K8192": (1, 8, 128, 256, 1),
    "M15N8192K14848": (1, 4, 128, 256, 8),
    "M15N2560K8192": (1, 4, 64, 256, 8),
    "M15N8192K2048": (1, 8, 128, 256, 1),
    "M15N14848K8192": (1, 8, 128, 256, 2),
    "M15N8192K7424": (1, 4, 64, 256, 4),
    "M15N1280K8192": (1, 4, 64, 256, 8),
    "M15N8192K1024": (2, 4, 64, 256, 1),
    "M15N7424K8192": (1, 4, 128, 256, 4),
    "M15N8192K3712": (1, 8, 64, 128, 1),
    "M16N10240K8192": (1, 16, 128, 256, 1),
    "M16N8192K8192": (1, 16, 128, 256, 1),
    "M16N59392K8192": (1, 8, 128, 256, 1),
    "M16N8192K29696": (1, 4, 128, 256, 10),
    "M16N5120K8192": (1, 4, 64, 256, 4),
    "M16N8192K4096": (1, 8, 128, 256, 1),
    "M16N29696K8192": (1, 8, 128, 256, 1),
    "M16N8192K14848": (1, 4, 128, 256, 8),
    "M16N2560K8192": (1, 4, 64, 256, 8),
    "M16N8192K2048": (1, 8, 128, 256, 1),
    "M16N14848K8192": (1, 8, 128, 256, 2),
    "M16N8192K7424": (1, 4, 64, 256, 4),
    "M16N1280K8192": (1, 4, 64, 256, 8),
    "M16N8192K1024": (1, 4, 64, 256, 1),
    "M16N7424K8192": (1, 4, 128, 256, 4),
    "M16N8192K3712": (1, 8, 64, 128, 1),
}

awq_w4a16_gemm_config_dict_80_gfx936 = {
    "llama2":llama2_awq_w4a16_gemm_config_dict_80_gfx936,
    "qwen25_72b":qwen25_72b_awq_w4a16_gemm_config_dict_80_gfx936
}


def multi_test(mode_name, G, dtype, use_marlin, use_excel=False):
    if use_marlin == False:
        raise ValueError("use_marlin must be True")
    # M_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    M_list = [17, 32,64]
    N_list = model_shape_dict[mode_name]["N"]
    K_list = model_shape_dict[mode_name]["K"]

    results = []
    for M in M_list:
        for i in range(len(N_list)):
            N = N_list[i]
            K = K_list[i]
            # config = awq_w4a16_gemm_config_dict_80_gfx936[mode_name][f"M16N{N}K{K}"]
            result = single_test(M=M,N=N,K=K,G=G, test_speed=True, use_random=True, dtype=torch.float16, use_marlin=use_marlin)
            if result is not None:
                results.append(result)

    # 如果 use_excel 为 True，保存结果到 Excel
    if use_excel and results:
        # 创建 DataFrame
        df = pd.DataFrame(results)

        # 重新排列列的顺序
        columns_order = ['M', 'N', 'K', 'G', 'correct', 'lightop_elapsed_time', 'awq_elapsed_time', 'fp16_elapsed_time',
                        'lightop_bandwidth', 'awq_bandwidth', 'fp16_bandwidth']
        df = df[columns_order]

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_awq_w4a16_gemm_results_{mode_name}_{timestamp}.xlsx"

        # 创建 Excel writer
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 写入数据（表格内不带单位）
            df.to_excel(writer, sheet_name='Results', index=False)

            # 获取工作簿和工作表
            workbook = writer.book
            worksheet = writer.sheets['Results']

            # 设置表头（带单位）
            headers_with_units = [
                'M', 'N', 'K', 'G', 'correct',
                'lightop_elapsed_time (us)', 'awq_elapsed_time (us)', 'fp16_elapsed_time (us)',
                'lightop_bandwidth (GB/s)', 'awq_bandwidth (GB/s)', 'fp16_bandwidth (GB/s)'
            ]

            for col_num, header in enumerate(headers_with_units, 1):
                worksheet.cell(row=1, column=col_num, value=header)

        print(f"结果已保存到 Excel 文件: {filename}")



def single_deal():
    #num_n_warps=1, num_k_warps=4, n_per_warp=32, k_per_warp=128, splitk=1
    # num_n_warps=1, num_k_warps=16, n_per_warp=128, k_per_warp=128, splitk=1
    M, N, K, G, dtype = 16, 8192,  3712, 128, torch.float16
    use_random, test_speed, use_marlin = True, True, True

    # single_test(M=M,N=N,K=K, G=G, test_speed=True, use_random=False, dtype=torch.float16, use_marlin=use_marlin)
    single_tunning(M=M,N=N,K=K, G=G, dtype=dtype, use_marlin=use_marlin)
    # sing_test_dequant(N=N,K=K, G=G, dtype=dtype, test_speed=test_speed, use_random=use_random, use_marlin=use_marlin)
    # num_n_warps=1
    # num_k_warps=8
    # n_per_warp=128
    # k_per_warp=256
    # splitk=1
    # _ = single_test_with_config(M=M,N=N,K=K,G=G, num_n_warps=num_n_warps, num_k_warps=num_k_warps, 
    # n_per_warp=n_per_warp, k_per_warp=k_per_warp, splitk=splitk, dtype=dtype, data_dict=None, use_random=use_random, test_speed=test_speed, use_marlin=use_marlin, mode="test")
    # n_per_warp=n_per_warp, k_per_warp=k_per_warp, splitk=splitk, dtype=dtype, data_dict=None, use_random=use_random, test_speed=test_speed, use_marlin=True, mode="test")
    # _ = single_test_with_config(M=M,N=N,K=K,G=G, num_n_warps=num_n_warps, num_k_warps=num_k_warps, 
    # n_per_warp=n_per_warp, k_per_warp=k_per_warp, splitk=splitk, dtype=dtype, data_dict=None, use_random=use_random, test_speed=test_speed, use_marlin=False, mode="test")








def multi_deal():
    use_random, test_speed, use_marlin, use_excel = True, True, True, True
    # mode_name = "dpsk_671b"
    mode_name = "qwen25_72b"
    G = 64
    dtype = torch.float16
    # multi_tunning(mode_name=mode_name, G=G, dtype=dtype, use_marlin=use_marlin)
    # multi_test(mode_name=mode_name, G=G, dtype=dtype, use_marlin=use_marlin)
    # multi_test_dequant(mode_name=mode_name, G=G, dtype=dtype, test_speed=test_speed, use_random=use_random, use_marlin=use_marlin)
    multi_test(mode_name=mode_name, G=G, dtype=dtype, use_marlin=use_marlin, use_excel=use_excel)
if __name__ == "__main__":
    # single_deal()
    multi_deal()
