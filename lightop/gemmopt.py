import torch
from lightop import op
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