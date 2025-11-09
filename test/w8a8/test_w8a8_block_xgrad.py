from typing import Optional, Type, Tuple, List
import math, time
import torch
import triton
import triton.language as tl

import pdb
torch.set_printoptions(precision=4, profile="default", sci_mode=False)

from lightop import gemm_w8a8_xgrad_asm


@triton.jit
def _per_token_group_quant_int8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Collums of input
    N,
    # Avoid to divide zero
    eps,
    # Information for int8
    int8_min,
    int8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform
    per-token-group quantization on a tensor.
    This function converts the tensor values into int8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / int8_max
    y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    M = x.numel() // group_size
    N = group_size
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    BLOCK = triton.next_power_of_2(N)#N是blocksize[1]
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
  
    _per_token_group_quant_int8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        int8_min=int8_min,
        int8_max=int8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s, BLOCK, num_warps, num_stages, M

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def _int8_gemm_helper(
    m: int,
    n: int,
    k: int,
    out_dtype: Type[torch.dtype] = torch.float16,
    device: str = "cuda",
    block_size: List[int]=[128, 128],
    best_config:Optional[list] = None):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    input = (torch.randn((m, k), device=device) * 5).to(dtype=out_dtype)
    weight = to_int8(torch.randn((n ,k), device=device) * 5)

    weight_scale = (torch.randn((math.ceil(n/block_size[0]), math.ceil(k/block_size[1])), device=device,
                           dtype=torch.float32))
    print("input.dtype:",input.dtype)
    #print("m:{} n:{} k:{},weight_scale.shape:{}".format(m,n,k,weight_scale.shape))

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]
    
    q_input, x_scale,_,_,_,_ = per_token_group_quant_int8(input_2d, block_size[1])
    
    return q_input, x_scale,weight,weight_scale


def native_w8a8_block_int8_matmul(A, B, As, Bs, block_size, output_dtype=torch.bfloat16):

    A = A.to(torch.float32)
    B = B.to(torch.float32)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N,)
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0]
    assert k_tiles == Bs.shape[1]

    C_shape = (M, N)
    C = torch.(C_shape, dtype=torch.float32, device=A.device)

    A_tiles = [A[:, i * block_k : min((i + 1) * block_k, K)] for i in range(k_tiles)]
    B_tiles = [
        [
            B[
                j * block_n : min((j + 1) * block_n, N),
                i * block_k : min((i + 1) * block_k, K),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]
    C_tiles = [C[:, j * block_n : min((j + 1) * block_n, N)] for j in range(n_tiles)]
    As_tiles = [As[:, i : i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C

def w8a8_block_int8_linear(
    m: int,
    n: int,
    k: int,
    out_dtype: Type[torch.dtype] = torch.bfloat16,
    device: str = "cuda",
    block_size: List[int]=[128, 128],
    bias: Optional[torch.Tensor] = None,
    best_config:Optional[dict]=None):
    
    
    q_input, x_scale,weight,weight_scale=_int8_gemm_helper(m=m,n=n,k=k,out_dtype=out_dtype,device=device,block_size=block_size)
    # x_scale = torch.ones_like(x_scale)
    # weight_scale = torch.ones_like(weight_scale)
    print(f"q_input:{q_input.shape}, x_scale:{x_scale.shape}, weight:{weight.shape}, weight_scale:{weight_scale.shape}")
    
    torch_output = native_w8a8_block_int8_matmul(q_input, weight, x_scale, weight_scale, block_size, out_dtype)

    x_scale = x_scale.permute(1, 0).contiguous()
    weight = weight.permute(1, 0).contiguous()
    weight_scale = weight_scale.permute(1,0).contiguous()
    output = gemm_w8a8_xgrad_asm(
        q_input, weight, x_scale, weight_scale, block_size, out_dtype
    )
    # print("output:", output.shape, output)
    # print("q_input:", q_input.shape)
    # print("weight:", weight.shape)
    print("x_scale:", x_scale.shape)
    print("weight_scale:", weight_scale.shape)

    if not torch.allclose(output, torch_output, rtol=1e-2, atol=5e-2):
        print("cuda 精度检查不合格！！！") 
        
    else:
        print("cuda 精度检查合格") 

if __name__ == "__main__":
    m1=[item if item < 17 else  1 << (item - 27) for item in range(1, 17)]
    m2=[item<<2  if item <17 else (item - 8)<<3  for item in range(5, 29)]
    m3=[2<<(item)  for item in range(7, 13)]
        
    m_list = m1+m2+m3
    n_list=[576,576,2048,7168,256,7168,1536,1536,2304,7168]
    k_list=[7168,7168,512,1024,7168,128,7168,1536,7168,1152]

    # m_list = [8192]
    # n_list=[7168]
    # k_list=[1152]

    block_size=[128, 128]
    
    out_dtype=torch.float16
    for i in range(0,len(k_list),1):
        for m in m_list:
            m, n, k = m, n_list[i], k_list[i]
            print("m:{} n:{} k:{} ".format(m, n, k))
            w8a8_block_int8_linear(m=m, n=n_list[i], k=k_list[i], block_size=block_size, out_dtype=out_dtype)