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