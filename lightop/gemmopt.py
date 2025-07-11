import torch
from typing import Optional
from lightop import op
import functools

w8a8_smooth_quant_config_dict = {
    "M1N3584K5120": (2, 16, 256, 1, False),
    "M2N3584K5120": (2, 16, 256, 1, False),
    "M3N3584K5120": (2, 16, 256, 1, False),
    "M4N3584K5120": (4, 16, 256, 1, False),
    "M5N3584K5120": (4, 16, 256, 1, False),
    "M6N3584K5120": (4, 16, 256, 1, False),
    "M7N3584K5120": (2, 32, 256, 2, False),
    "M8N3584K5120": (2, 32, 256, 2, False),
    "M9N3584K5120": (2, 32, 256, 2, False),
    "M10N3584K5120": (2, 32, 256, 2, False),
    "M11N3584K5120": (2, 32, 256, 2, False),
    "M12N3584K5120": (2, 32, 256, 2, False),
    "M13N3584K5120": (2, 32, 256, 2, False),
    "M14N3584K5120": (2, 32, 256, 2, False),
    "M15N3584K5120": (2, 32, 256, 2, False),
    "M16N3584K5120": (4, 32, 128, 2, False),
    "M1N5120K2560": (1, 16, 256, 1, False),
    "M2N5120K2560": (1, 16, 256, 1, False),
    "M3N5120K2560": (1, 16, 256, 1, True),
    "M4N5120K2560": (1, 16, 256, 1, False),
    "M5N5120K2560": (1, 16, 256, 1, True),
    "M6N5120K2560": (1, 16, 256, 1, False),
    "M7N5120K2560": (1, 16, 256, 1, False),
    "M8N5120K2560": (2, 32, 256, 1, False),
    "M9N5120K2560": (2, 32, 256, 1, False),
    "M10N5120K2560": (2, 32, 256, 1, False),
    "M11N5120K2560": (2, 32, 256, 1, False),
    "M12N5120K2560": (2, 32, 256, 1, False),
    "M13N5120K2560": (2, 32, 256, 1, False),
    "M14N5120K2560": (2, 32, 256, 1, False),
    "M15N5120K2560": (2, 32, 256, 1, False),
    "M16N5120K2560": (2, 32, 256, 1, False),
    "M1N5120K13824": (4, 16, 256, 1, False),
    "M2N5120K13824": (4, 16, 256, 1, False),
    "M3N5120K13824": (2, 16, 256, 1, False),
    "M4N5120K13824": (2, 16, 256, 1, False),
    "M5N5120K13824": (2, 16, 256, 1, False),
    "M6N5120K13824": (2, 16, 256, 1, False),
    "M7N5120K13824": (2, 16, 256, 1, False),
    "M8N5120K13824": (2, 16, 256, 1, False),
    "M9N5120K13824": (2, 16, 256, 1, False),
    "M10N5120K13824": (2, 16, 256, 1, False),
    "M11N5120K13824": (4, 32, 256, 1, False),
    "M12N5120K13824": (4, 32, 256, 1, False),
    "M13N5120K13824": (4, 32, 256, 1, False),
    "M14N5120K13824": (4, 32, 256, 1, False),
    "M15N5120K13824": (4, 32, 256, 1, False),
    "M16N5120K13824": (4, 32, 256, 1, False),
    "M1N27684K5120": (8, 32, 256, 10, False),
    "M2N27684K5120": (8, 32, 256, 10, False),
    "M3N27684K5120": (16, 16, 256, 1, False),
    "M4N27684K5120": (16, 16, 256, 1, False),
    "M5N27684K5120": (16, 16, 256, 1, False),
    "M6N27684K5120": (16, 16, 256, 1, False),
    "M7N27684K5120": (16, 16, 256, 1, False),
    "M8N27684K5120": (16, 16, 256, 1, False),
    "M9N27684K5120": (16, 16, 256, 1, False),
    "M10N27684K5120": (1, 64, 256, 1, True),
    "M11N27684K5120": (1, 64, 256, 1, True),
    "M12N27684K5120": (1, 64, 256, 1, True),
    "M13N27684K5120": (1, 64, 256, 1, True),
    "M14N27684K5120": (2, 64, 256, 1, True),
    "M15N27684K5120": (2, 64, 256, 1, True),
    "M16N27684K5120": (4, 32, 256, 1, False),
}

@functools.lru_cache(maxsize=128)
def w8a8_smooth_quant_config(m, n, k):
    """
    根据输入维度(m, n, k)返回最优的配置参数
    
    Args:
        m (int): 输入矩阵的行数
        n (int): 输出矩阵的列数  
        k (int): 输入矩阵的列数/权重矩阵的行数
        
    Returns:
        dict: 包含warps, nperblock, kperwarp, splitk的最优配置
    """
    # 基于tuning结果的最优配置字典
    # value为元组：(warps, nperblock, kperwarp, splitk, time_ms)
    
    # 查找匹配的配置
    key = f"M{m}N{n}K{k}"
    if key in w8a8_smooth_quant_config_dict:
        warps, nperblock, kperwarp, splitk, use_n_warps = w8a8_smooth_quant_config_dict[key]
        return warps, nperblock, kperwarp, splitk, use_n_warps
    else:
        # 如果没有完全匹配的配置，返回默认配置，并提示警告
        print(f"警告：未找到维度({m}, {n}, {k})的最优配置，已返回默认配置 (2, 32, 256, 1, False)。如需最佳性能，请先进行tuning。")
        return 2, 32, 256, 1, False

def gemm_w8a8_smooth(a,b,scale_a,scale_b,bias,out_dtype):
    assert a.dim()==2 and b.dim()==2 and a.is_contiguous() and b.is_contiguous()==False
    m = a.shape[0]
    n = b.shape[1] # b的shape是(n,k)
    k = a.shape[1]
    warps, nperblock, kperwarp, splitk, use_n_warps = w8a8_smooth_quant_config(m, n, k)
    return op.gemm_w8_a8_smooth_colmajor(a, b, scale_a, scale_b, bias, out_dtype, warps, nperblock, kperwarp, splitk, use_n_warps)


def gemm(a,b):
    assert a.dim()==2 and b.dim()==2 and a.is_contiguous()
    return op.gemm_opt(a,b)

def gemm_w4a16(a,b,scale_n,zero_n):
    assert a.dim()==2 and b.dim()==2 and a.is_contiguous() and b.is_contiguous() == False
    return op.gemm_w4_a16(a,b,scale_n,zero_n)

def w4tow16(a,scale_n,zero_n):
    return op.w4_to_w16(a,scale_n,zero_n)

def gemm_w8a16_fp8(a,b,scale,warps=0,n_per_block=0,splitk=0):
    assert a.dim()==2 and b.dim()==2 and a.is_contiguous() and b.is_contiguous()==False
    return op.gemm_w8_a16_fp8_colmajor(a,b,scale,warps,n_per_block,splitk)

def fp8_to_16(out,intput,scale):
    return op.fp8to16(out,intput,scale)

def moe_gemm_w8a8(input: torch.Tensor,
                    b_qweight: torch.Tensor,
                    output: torch.Tensor,
                    a_scale: torch.Tensor,
                    b_scale: torch.Tensor,
                    topk_weights: Optional[torch.Tensor],
                    sorted_token_ids: torch.Tensor,
                    experts_ids: torch.Tensor,
                    num_tokens_post_pad: torch.Tensor, 
                    top_k: int,
                    mode: int) -> torch.Tensor:
    op.moe_gemm_w8a8(input, b_qweight, output, a_scale, b_scale, topk_weights,
                                    sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode)
    return output

