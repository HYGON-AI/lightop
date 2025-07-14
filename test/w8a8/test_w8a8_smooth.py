import torch
from lightop import op
import time
import lmslimquant
from lmslim import quant_ops 
from typing import Optional
import lightop

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import functools


@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
):
    # Adapted from https://github.com/InternLM/lmdeploy/blob/086481ed84b59bee3b8e4274e5fc69620040c048/lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py#L282
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask,
                other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = libdevice.nearbyint(x_q).to(tl.int8)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def per_token_quant_int8(x):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1, ),
                         device=x.device,
                         dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)

    assert x.is_contiguous()
    _per_token_quant_int8[(M, )](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

    return x_q, scales


torch.set_printoptions(precision=4,threshold=100000,linewidth=140,sci_mode=False)
torch.random.manual_seed(0)
repeat_num=100

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def _rocblas_scaled_mm(a: torch.Tensor,   #input
                            b: torch.Tensor,  #weight
                            scale_a: torch.Tensor,
                            scale_b: torch.Tensor,
                            out_dtype: torch.dtype,
                            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return quant_ops.rocblas_scaled_mm_nn(a,b,scale_a,scale_b,out_dtype,bias)

def benchmark_w8a8_operator(a, b, scale_a, scale_b, bias, out_dtype, include_quant=True, warmup_iters=10, test_iters=100):
    """
    对w8a8算子进行速度测试
    
    Args:
        include_quant: 是否包含量化过程在测试中
        warmup_iters: warmup次数
        test_iters: 测试次数
    """
    device = a.device
    
    # Warmup阶段
    # print(f"开始warmup {warmup_iters}次...")
    for _ in range(warmup_iters):
        if include_quant:
            # 包含量化的测试
            a_quant, scale_a_quant = per_token_quant_int8(a)
            _ = lightop.gemm_w8a8_smooth(a,b,scale_a,scale_b,bias,out_dtype)
        else:
            # 不包含量化的测试
            _ = lightop.gemm_w8a8_smooth(a,b,scale_a,scale_b,bias,out_dtype)
    
    # 同步GPU
    torch.cuda.synchronize()
    
    # 性能测试阶段
    # print(f"开始性能测试 {test_iters}次...")
    start_time = time.time()
    
    if include_quant:
        # 包含量化的测试
        for _ in range(test_iters):
            a_quant, scale_a_quant = per_token_quant_int8(a)
            _ = lightop.gemm_w8a8_smooth(a,b,scale_a,scale_b,bias,out_dtype)
    else:
        # 不包含量化的测试
        for _ in range(test_iters):
            _ = lightop.gemm_w8a8_smooth(a,b,scale_a,scale_b,bias,out_dtype)
    
    # 同步GPU并计算平均耗时
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time_us = (end_time - start_time) * 1e6  # 转换为微秒
    avg_time_us = total_time_us / test_iters
    
    test_type = "包含量化" if include_quant else "不包含量化"
    print(f"w8a8算子速度测试结果 ({test_type}) - M{m}xN{n}xK{k}: 总耗时: {total_time_us:.2f} us, 平均耗时: {avg_time_us:.2f} us, 测试次数: {test_iters}")
    
    return avg_time_us
 

def main(m, n, k, use_random=True, use_bias=False, out_dtype=torch.bfloat16, use_quant=True, enable_speed_test=False, speed_test_include_quant=False):
    # 这里可以添加测试代码
    device = "cuda"
    
    if use_random == False:
        a = torch.ones((m, k), device=device, dtype=torch.float16)
        if use_quant:
            a, scale_a = per_token_quant_int8(a)
        else:
            a = to_int8(a)
            scale_a = (torch.ones((m, 1), device=device,dtype=torch.float32))
        b = to_int8(torch.ones((n, k), device=device).t())
        scale_b = (torch.ones((n,1), device=device,
                            dtype=torch.float32))
        if use_bias:
            bias = torch.zeros((n,), device=device, dtype=out_dtype)
        else:
            bias = None
    else:
        a = torch.randn((m, k), device=device, dtype=torch.float16) * 5
        if use_quant:
            a, scale_a = per_token_quant_int8(a)
        else:
            a = to_int8(a)
            scale_a = (torch.randn((m, 1), device=device,
                           dtype=torch.float32))
        b = to_int8(torch.randn((n, k), device=device).t() * 5)
        scale_b = (torch.randn((n,1), device=device,
                           dtype=torch.float32))
        if use_bias:
            bias = torch.randn((n,), device=device, dtype=out_dtype) * 10
        else:
            bias = None    
    # 保存原始输入用于速度测试
    a_original = a.clone() if not use_quant else torch.randn((m, k), device=device, dtype=torch.float16) * 5
    scale_a_original = scale_a.clone()
    
    out = _rocblas_scaled_mm(a,b.contiguous().reshape(k,-1),scale_a,scale_b,out_dtype,bias)
    status, out_w8a8 = lightop.gemm_w8a8_smooth(a,b,scale_a,scale_b,bias,out_dtype)
    if status is False:
        print("gemm_w8a8_smooth 执行失败，错误码:", status)
        return
    print("rocblas out:",out[0:1,0:1])
    print("w8a8 out:",out_w8a8[0:1,0:1])
    diff_max = (out - out_w8a8).abs().max()
    if not (torch.allclose(out, out_w8a8, rtol=1e-2, atol=5e-2)):
        print("error gemm_w8a8_smooth 精度检查不合格！！！")  
    else:
        print("success gemm_w8a8_smooth 精度检查合格")    
    print(f"diff_max={diff_max}")
    
    # 速度测试
    if enable_speed_test:
        print("\n" + "="*50)
        print("开始w8a8算子速度测试")
        print("="*50)
        
        # 测试包含量化的情况
        if speed_test_include_quant:
            benchmark_w8a8_operator(a_original, b, scale_a_original, scale_b, bias, 
                                  out_dtype, include_quant=True, warmup_iters=10, test_iters=100)
        
        # 测试不包含量化的情况
        benchmark_w8a8_operator(a, b, scale_a, scale_b, bias, 
                              out_dtype, include_quant=False, warmup_iters=10, test_iters=100)



if __name__ == "__main__":
    m1=[item if item < 17 else  1 << (item - 27) for item in range(1, 17)]
    m2=[item<<2  if item <17 else (item - 8)<<3  for item in range(5, 29)]
    m3=[2<<(item)  for item in range(7, 15)]

    m_list = m1+m2+m3
    # n_list=[1280, 4096, 384, 4096, 768, 4096, 192, 4096,]
    # k_list=[4096, 1024, 4096, 192,4096,512,4096,96]
    # m_list = [16384]
    n_list = [7168, 256, 7168, 2048, 1536, 1536, 7168, 2304, 7168,  576]
    k_list = [128, 7168, 1024, 512, 1536, 7168, 1152, 7168, 1024, 7168]
    for i in range(0,len(k_list),1):
        for m in m_list:
            m, n, k = m, n_list[i], k_list[i]
            print("m:{} n:{} k:{} ".format(m, n, k))
            use_bias = False
            use_random = True
            # out_dtype = torch.bfloat16
            out_dtype = torch.float16
            use_quant = True
            enable_speed_test = False  # 设置为True启用速度测试
            speed_test_include_quant = False  # 速度测试是否包含量化过程
            main(m, n, k, use_random, use_bias, out_dtype, use_quant, enable_speed_test, speed_test_include_quant)
