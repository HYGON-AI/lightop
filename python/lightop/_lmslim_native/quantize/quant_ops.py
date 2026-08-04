from importlib import import_module


class _LazyLmslimQuant:
    """Load the optional legacy native extension only when an API needs it."""

    def __getattr__(self, name):
        module = import_module("lmslimquant")
        globals()["lmslimquant"] = module
        return getattr(module, name)


lmslimquant = _LazyLmslimQuant()
import torch
import torch.nn.functional as F
from typing import Optional, List
from ..layers.gemm.int8_utils import matmul_int8
from lightop import envs as lsenvs
# from vllm.utils.torch_utils import direct_register_custom_op
from ..vllm_compat.direct_register_custom_op import direct_register_custom_op

from torch.library import Library
lmslim_lib = Library("lmslim", "FRAGMENT")
from .utils.awq_gemm_get_best_template import AWQ_W4A16_config_cuda
use_lightop = lsenvs.LMSLIM_USE_LIGHTOP
device_name = lsenvs.LMSLIM_GPU_NAME

if use_lightop:
    from ...gemmopt import gemm_w8a8_smooth

# from lightop import op as lightop_op
from ...op import rms_norm_dynamic_per_token_quant
from ...fuse_silu_mul_quant import fuse_silu_mul_quant


def get_awq_workspace_size(N_max : int, K_max : int, GroupSize:int)-> int:
    return lmslimquant.get_awq_workspace_size(N_max, K_max, GroupSize)

def awq_gemm(input: torch.Tensor, weight: torch.Tensor,
             zeros_and_scale:torch.Tensor,
             m:int,n:int,k:int,
             group_size:int,padding_group:int,splikspace:torch.Tensor,
            splikspacesize:int, config=None) -> torch.Tensor:

    input_tersor = input
    #初始化padding_spacesize
    padding_spacesize = 0
    pad_k = k
    #判断是否有pad
    if padding_group > 0:
        pad = group_size * padding_group
        pad_k += pad

    if m <= 512:
        if not config:
            cur_config = AWQ_W4A16_config_cuda(m, n, k, group_size)
        else:
            cur_config = config


        if padding_group > 0:
            padding_spacesize = m * pad_k * 2

            #最多给1/2用于加pad
            if splikspacesize > 2 * padding_spacesize:
                input_padded = splikspace[-m * pad_k:]
                lmslimquant.awq_input_padding(input,m,k,pad_k,input_padded,padding_spacesize)
                input_padded = input_padded.view(m,pad_k)
            else:
                padding_spacesize = 0
                input_padded = torch.cat(
                    (input, torch.zeros((m, pad), dtype=input.dtype, device=input.device)),
                    dim=1).contiguous()
            input_tersor = input_padded

        output = lmslimquant.gemm_w4a16_v2(input_tersor,
                        weight,
                        zeros_and_scale,
                        #output,
                        m,
                        n,
                        k,
                        pad_k,          #StrideA
                        pad_k,          #StrideB
                        pad_k,          #StrideB_padded
                        n,              #StrideC
                        group_size,
                        splikspace,
                        int(splikspacesize-padding_spacesize),
                        cur_config
                        )

    else:
        if n >= 38400:
            chunk_size = 512
            outputs = []
            for i in range(0, m, chunk_size):
                chunk_m = min(chunk_size, m - i)
                input_chunk = input_tersor[i:i+chunk_m]
                # 递归调用 awq_gemm，每个块的 m <= 512，会走第一个分支
                output_chunk = awq_gemm(input_chunk, weight, zeros_and_scale,
                                       chunk_m, n, k, group_size, padding_group,
                                       splikspace, splikspacesize, config)
                outputs.append(output_chunk)
            output = torch.cat(outputs, dim=0)

        else:
            dequant_weight_size = n * k * 2

            if dequant_weight_size > splikspacesize-padding_spacesize:
                dqweight = torch.empty((n, k), dtype=torch.float16).cuda()
            else:
                dqweight = splikspace[:n * k]
            lmslimquant.awq_dequant_colmajor(dqweight,
                                            weight,
                                            zeros_and_scale,
                                            k,
                                            pad_k, #StrideB_padded
                                            n,
                                            group_size,
                                            dequant_weight_size)
            output = F.linear(input_tersor,dqweight.view(n,k))
            if dequant_weight_size <= splikspacesize-padding_spacesize:
                dqweight.zero_()

    if padding_group > 0 and padding_spacesize != 0:
        input_padded.zero_()

    return output


def gptq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    return torch.ops.lmslim.gptq_gemm1(a,b_q_weight,b_gptq_qzeros,b_gptq_scales,b_g_idx,use_exllama,bit)

def gptq_gemm1(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    return lmslimquant.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                                  b_g_idx, use_exllama, bit)

def gptq_gemm1_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    M = a.size(0)
    N =b_q_weight.size(1)
    return torch.empty((M, N), dtype=a.dtype, device=a.device)

direct_register_custom_op(
    op_name="gptq_gemm1",
    op_func=gptq_gemm1,
    mutates_args=[],
    target_lib=lmslim_lib,
    fake_impl=gptq_gemm1_fake,
)

def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor,
                 bit: int) -> None:
    lmslimquant.gptq_shuffle(q_weight, q_perm, bit)

def convert_s4(qw: torch.Tensor, qz: torch.Tensor, s: torch.Tensor,
               group_size: int):
    assert qw.is_contiguous()
    assert qz.is_contiguous()
    assert s.is_contiguous()
    _qw = torch.zeros_like(qw)
    _sz = torch.zeros_like(s, dtype=torch.int32)  # half2
    _ws = torch.zeros_like(s)
    lmslimquant.convert_s4_k_m8(_qw, _sz, _ws, qw, s, qz,
                        qw.size(-1) * 8, qw.size(0), group_size)
    return _qw, _sz

def sz_permute(sz:torch.Tensor):
    dim0=sz.shape[0]
    n=sz.shape[1]
    _sz=torch.zeros_like(sz)
    lmslimquant.permute_trans(_sz,sz,dim0,n)
    _sz=_sz.reshape(n,-1)
    return _sz

def awq_dequant_colmajor(qweight:torch.Tensor,
                        zeros_and_scale:torch.Tensor,
                        k:int,
                        k_pad:int,
                        n:int,
                        group_size:int,
                        splikspace:torch.Tensor,
                        splikspacesize:int
                        )->torch.Tensor:
    # dqweight= torch.empty((n,k),dtype=torch.float16).cuda()
    lmslimquant.awq_dequant_colmajor(splikspace,
                                         qweight,
                                         zeros_and_scale,
                                         k,
                                         k_pad,
                                         n,
                                         group_size,
                                         int(splikspacesize))
    return splikspace[:n*k].view(n,k)

def dequant_w4_gemm_colmajor(qweight:torch.Tensor,
                                zeros_and_scale:torch.Tensor,
                                k:int,
                                n:int,
                                group_size:int
                             )->torch.Tensor:
    dqweight= torch.zeros((n,k),dtype=torch.float16).cuda()
    lmslimquant.dequant_w4_gemm_colmajor(dqweight,
                                         qweight,
                                         zeros_and_scale,
                                         k,
                                         n,
                                         group_size)
    return dqweight

def dequant_w4_gemm_rowmajor(qweight:torch.Tensor,
                                zeros_and_scale:torch.Tensor,
                                k:int,
                                n:int,
                                group_size:int
                             )->torch.Tensor:
    dqweight= torch.zeros((k,n),dtype=torch.float16).cuda()
    lmslimquant.dequant_w4_gemm_rowmajor(dqweight,
                                         qweight,
                                         zeros_and_scale,
                                         k,
                                         n,
                                         group_size)
    return dqweight

def GetAWQShareWorkspaceSize()->int:
    return 2<<27

def GetAWQShareWorkspace()->torch.Tensor:
    WorkspaceSize=GetAWQShareWorkspaceSize()
    return torch.zeros(WorkspaceSize // 2 + 1, dtype=torch.float16).cuda()

def rocblas_sgemm_fp16_tn(a: torch.Tensor,
                    b: torch.Tensor,
                    out_dtype: torch.dtype,
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    gemmoutput=lmslimquant.rocblas_gemm_fp16(a,b,True)
    output=gemmoutput.to(out_dtype)
    if bias is not None:
        output = output + bias.to(out_dtype)
    return output

def rocblas_sgemm_fp16_nn(a: torch.Tensor,
                    b: torch.Tensor,
                    out_dtype: torch.dtype,
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    gemmoutput=lmslimquant.rocblas_gemm_fp16(a,b,False)
    output=gemmoutput.to(out_dtype)
    if bias is not None:
        output = output + bias.to(out_dtype)
    return output

def rocblas_sgemm_int8_tn(a: torch.Tensor,
                    b: torch.Tensor,
                    out_dtype: torch.dtype,
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    gemmoutput=lmslimquant.rocblas_gemm_int8(a,b,True)
    output=gemmoutput.to(out_dtype)
    if bias is not None:
        output = output + bias
    return output

def rocblas_sgemm_int8_nn(a: torch.Tensor,
                    b: torch.Tensor,
                    out_dtype: torch.dtype,
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    gemmoutput=lmslimquant.rocblas_gemm_int8(a,b,False)
    output=gemmoutput.to(out_dtype)
    if bias is not None:
        output = output + bias
    return output




def _rocblas_scaled_mm_tn(a: torch.Tensor,   #input
                            b: torch.Tensor,  #weight
                            scale_a: torch.Tensor,
                            scale_b: torch.Tensor,
                            out_dtype: torch.dtype,
                            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    #scale_a.shape=[m,1]
    #scale_a.shape=[n,1]
    #a.shape=[m,k]
    #b.shape=[k,n]
    scales= scale_a * scale_b.T
    try:
        gemmoutput = lmslimquant.rocblas_gemm_int8(a, b, True)
    except ModuleNotFoundError as exc:
        if exc.name != "lmslimquant":
            raise
        try:
            gemmoutput = torch._int_mm(a, b)
        except RuntimeError:
            gemmoutput = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    output=(scales*gemmoutput.to(dtype=torch.float32)).to(out_dtype)

    if bias is not None:
        output = output + bias

    return output


def _rocblas_scaled_mm_nn(a: torch.Tensor,   #input
                            b: torch.Tensor,  #weight
                            scale_a: torch.Tensor,
                            scale_b: torch.Tensor,
                            out_dtype: torch.dtype,
                            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    #scale_a.shape=[m,1]
    #scale_a.shape=[n,1]
    #a.shape=[m,k]
    #b.shape=[k,n]
    scales= scale_a * scale_b.T
    try:
        gemmoutput = lmslimquant.rocblas_gemm_int8(a, b, False)
    except ModuleNotFoundError as exc:
        if exc.name != "lmslimquant":
            raise
        try:
            gemmoutput = torch._int_mm(a, b)
        except RuntimeError:
            gemmoutput = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    output=(scales*gemmoutput.to(dtype=torch.float32)).to(out_dtype)

    if bias is not None:
        output = output + bias

    return output

def rocblas_scaled_mm_tn_fake(a: torch.Tensor,
                         b: torch.Tensor,
                         scale_a: torch.Tensor,
                         scale_b: torch.Tensor,
                         out_dtype: torch.dtype,
                         bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    M = a.size(0)
    N = b.size(1)
    return torch.empty((M, N), dtype=out_dtype, device=a.device)

direct_register_custom_op(
    op_name="_rocblas_scaled_mm_tn",
    op_func=_rocblas_scaled_mm_tn,
    mutates_args=[],
    target_lib=lmslim_lib,
    fake_impl=rocblas_scaled_mm_tn_fake,
)

def rocblas_scaled_mm_nn_fake(a: torch.Tensor,   #input
                         b: torch.Tensor,  #weight
                         scale_a: torch.Tensor,
                         scale_b: torch.Tensor,
                         out_dtype: torch.dtype,
                         bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    M = a.size(0)
    N = b.size(1)
    return torch.empty((M, N), dtype=out_dtype, device=a.device)

direct_register_custom_op(
    op_name="_rocblas_scaled_mm_nn",
    op_func=_rocblas_scaled_mm_nn,
    mutates_args=[],
    target_lib=lmslim_lib,
    fake_impl=rocblas_scaled_mm_nn_fake,
)

if use_lightop:
    def lightop_channel_int8_mm(a: torch.Tensor,
                            b: torch.Tensor,
                            scale_a: torch.Tensor,
                            scale_b: torch.Tensor,
                            out_dtype: torch.dtype,
                            bias: Optional[torch.Tensor] = None) -> tuple[bool, torch.Tensor]:
            return gemm_w8a8_smooth(a, b,scale_a,scale_b,bias,out_dtype)

    def lightop_channel_int8_mm_fake(a: torch.Tensor,
                                 b: torch.Tensor,
                                 scale_a: torch.Tensor,
                                 scale_b: torch.Tensor,
                                 out_dtype: torch.dtype,
                                 bias: Optional[torch.Tensor] = None) -> tuple[bool, torch.Tensor]:
        M = a.size(0)
        N = b.size(1)
        return True, torch.empty((M, N), dtype=out_dtype, device=a.device)


    direct_register_custom_op(
        op_name="lightop_channel_int8_mm",
        op_func=lightop_channel_int8_mm,
        mutates_args=[],
        target_lib=lmslim_lib,
        fake_impl=lightop_channel_int8_mm_fake,
    )


def rocblas_scaled_mm_tn(a: torch.Tensor,   #input
                         b: torch.Tensor,  #weight
                         scale_a: torch.Tensor,
                         scale_b: torch.Tensor,
                         out_dtype: torch.dtype,
                         bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.lmslim._rocblas_scaled_mm_tn(a,b,scale_a,scale_b,out_dtype,bias)

def rocblas_scaled_mm_nn(a: torch.Tensor,   #input
                         b: torch.Tensor,  #weight
                         scale_a: torch.Tensor,
                         scale_b: torch.Tensor,
                         out_dtype: torch.dtype,
                         bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.lmslim._rocblas_scaled_mm_nn(a,b,scale_a,scale_b,out_dtype,bias)

def triton_scaled_mm(a: torch.Tensor,   #input
                            b: torch.Tensor,  #weight
                            scale_a: torch.Tensor,
                            scale_b: torch.Tensor,
                            out_dtype: torch.dtype,
                            bias: Optional[torch.Tensor] = None,
                            best_config:Optional[list] = None) -> torch.Tensor:
    output=None #初始输出
    status=False #初始状态
    if use_lightop and hasattr(torch.ops.lmslim, "lightop_channel_int8_mm"):
        status, output = torch.ops.lmslim.lightop_channel_int8_mm(a, b, scale_a, scale_b, out_dtype, bias)
    if not use_lightop or status==False:
        output = matmul_int8(a, scale_a, b, scale_b, out_dtype, best_config)
        if bias is not None:
            output += bias
    return output

# transpose_flag: support NT only now
# input_type: fp8/int8
# output_type: fp16/bf16
def hipblaslt_gemm_channelwise_wrapper(
        a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        out_dtype : torch.dtype,
        m : int,
        n : int,
        k : int,
        batch : int,
        transpose_flag : str,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    if a.dtype == torch.int8:
        alpha = torch.tensor(1, device='cpu', dtype=torch.int32)
        beta = torch.tensor(0, device='cpu', dtype=torch.int32)
    elif a.dtype == torch.float8_e4m3fn:
        alpha = torch.tensor(1, device='cpu', dtype=torch.float32)
        beta = torch.tensor(0, device='cpu', dtype=torch.float32)
    else:
        raise ValueError("Unsupported input dtype")
    return lmslimquant.hipblaslt_gemm_channelwise(
            a, b, scale_a, scale_b, out_dtype, m, n, k, batch, transpose_flag, alpha, beta, bias)

def hipblaslt_gemm_channelwise_fake(
        a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        out_dtype : torch.dtype,
        m : int,
        n : int,
        k : int,
        batch : int,
        transpose_flag : str,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    assert a.shape[-2:] == (m, k)
    assert b.shape[-2:] == (n, k)
    assert a.is_contiguous() and b.is_contiguous()
    return torch.empty((batch, n, m), dtype=out_dtype, device=a.device)

direct_register_custom_op(
    op_name = "hipblaslt_gemm_channelwise",
    op_func = hipblaslt_gemm_channelwise_wrapper,
    mutates_args = [],
    target_lib = lmslim_lib,
    fake_impl = hipblaslt_gemm_channelwise_fake,
)

# KME/gfx928 int8_bf16_cw: same signature as channelwise; C/D keep out_dtype, compute=fp32.
# transpose_flag: support NT only now (public API); wrapper accepts swapped TN like channelwise.
# input_type: fp8/int8
# output_type: fp16/bf16
def hipblaslt_gemm_channelwise_kme_wrapper(
        a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        out_dtype : torch.dtype,
        m : int,
        n : int,
        k : int,
        batch : int,
        transpose_flag : str,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    if a.dtype == torch.int8:
        alpha = torch.tensor(1, device='cpu', dtype=torch.int32)
        beta = torch.tensor(0, device='cpu', dtype=torch.int32)
    elif a.dtype == torch.float8_e4m3fn:
        alpha = torch.tensor(1, device='cpu', dtype=torch.float32)
        beta = torch.tensor(0, device='cpu', dtype=torch.float32)
    else:
        raise ValueError("Unsupported input dtype")
    return lmslimquant.hipblaslt_gemm_channelwise_kme(
            a, b, scale_a, scale_b, out_dtype, m, n, k, batch, transpose_flag, alpha, beta, bias)

def hipblaslt_gemm_channelwise_kme_fake(
        a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        out_dtype : torch.dtype,
        m : int,
        n : int,
        k : int,
        batch : int,
        transpose_flag : str,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    assert a.shape[-2:] == (m, k)
    assert b.shape[-2:] == (n, k)
    assert a.is_contiguous() and b.is_contiguous()
    return torch.empty((batch, n, m), dtype=out_dtype, device=a.device)

direct_register_custom_op(
    op_name = "hipblaslt_gemm_channelwise_kme",
    op_func = hipblaslt_gemm_channelwise_kme_wrapper,
    mutates_args = [],
    target_lib = lmslim_lib,
    fake_impl = hipblaslt_gemm_channelwise_kme_fake,
)

from enum import IntEnum
class BlockSize(IntEnum):
    block_128x128 = 1
    block_64x64 = 2

# not support batch now
# not support bias now
# transpose_flag: support NN only now
# input_type: fp8
# output_type: fp16/bf16
def hipblaslt_gemm_blockwise_wrapper(
        a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        out_dtype : torch.dtype,
        m : int,
        n : int,
        k : int,
        batch : int,
        transpose_flag : str,
        block_size : int,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    alpha = torch.tensor(1, device='cpu', dtype=torch.float32)
    beta = torch.tensor(0, device='cpu', dtype=torch.float32)
    return lmslimquant.hipblaslt_gemm_blockwise(
            a, b, scale_a, scale_b, torch.bfloat16, m, n, k, 1, 'NN', block_size, alpha, beta, bias)

def hipblaslt_gemm_blockwise_fake(
        a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        out_dtype : torch.dtype,
        m : int,
        n : int,
        k : int,
        batch : int,
        transpose_flag : str,
        block_size : int,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    assert a.is_contiguous() and b.is_contiguous()
    alpha = torch.tensor(1, device='cpu', dtype=torch.float32)
    beta = torch.tensor(0, device='cpu', dtype=torch.float32)
    return torch.empty((batch, n, m), dtype = out_dtype, device = a.device)

direct_register_custom_op(
    op_name = "hipblaslt_gemm_blockwise",
    op_func = hipblaslt_gemm_blockwise_wrapper,
    mutates_args = [],
    target_lib = lmslim_lib,
    fake_impl = hipblaslt_gemm_blockwise_fake,
)

def hipblaslt_w8a8_channelwise_gemm(
        a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        m : int,
        n : int,
        k : int,
        transpose_flag : str,
        out_dtype : torch.dtype,
        bias : Optional[torch.Tensor] = None) -> tuple[bool, torch.Tensor]:
    assert transpose_flag == 'NT'
    assert a.is_contiguous() and b.is_contiguous()
    assert scale_a.dtype is torch.float32 and scale_b.dtype is torch.float32
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.dtype == out_dtype
    assert a.shape[-1] == b.shape[-1]

    res = torch.ops.lmslim.hipblaslt_gemm_channelwise(
            b, a, scale_b, scale_a, out_dtype, n, m, k, b.shape[0] if b.dim() == 3 else 1, 'TN', bias)
    return True, res

def hipblaslt_w8a8_channelwise_gemm_kme(
        a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        m : int,
        n : int,
        k : int,
        transpose_flag : str,
        out_dtype : torch.dtype,
        bias : Optional[torch.Tensor] = None) -> tuple[bool, torch.Tensor]:
    assert transpose_flag == 'NT'
    assert a.is_contiguous() and b.is_contiguous()
    assert scale_a.dtype is torch.float32 and scale_b.dtype is torch.float32
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.dtype == out_dtype
    assert a.shape[-1] == b.shape[-1]

    res = torch.ops.lmslim.hipblaslt_gemm_channelwise_kme(
            b, a, scale_b, scale_a, out_dtype, n, m, k, b.shape[0] if b.dim() == 3 else 1, 'TN', bias)
    return True, res

def hipblaslt_w8a8_blockwise_gemm(
        a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        m : int,
        n : int,
        k : int,
        transpose_flag : str,
        out_dtype : torch.dtype,
        block_size : BlockSize,
        bias : Optional[torch.Tensor] = None) -> tuple[bool, torch.Tensor]:
    assert transpose_flag == 'NN'
    assert a.is_contiguous() and b.is_contiguous()
    assert scale_a.dtype is torch.float32 and scale_b.dtype is torch.float32
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.dtype == out_dtype
    assert a.shape[-1] == b.shape[-2]


    res = torch.ops.lmslim.hipblaslt_gemm_blockwise(
            b, a, scale_b, scale_a,
            out_dtype,
            n, m, k, b.shape[0] if b.dim() == 3 else 1,
            transpose_flag, block_size, bias)
    return True, res

def blaslt_w8a8_fp16_gemm_wrapper(a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        d : torch.Tensor,
        m : int,
        n : int,
        k : int,
        batch : int,
        transpose_flag : str,
        bias: Optional[torch.Tensor] = None
    ) -> None:
    alpha = torch.tensor(1, dtype=torch.int32, device='cpu')
    beta = torch.tensor(0, dtype=torch.int32, device='cpu')
    lmslimquant.blaslt_w8a8_fp16_gemm(a, b, scale_a, scale_b, d, m, n, k, batch, transpose_flag, alpha, beta, bias)
    return

def blaslt_w8a8_fp16_gemm_fake(a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        d : torch.Tensor,
        m : int,
        n : int,
        k : int,
        batch : int,
        transpose_flag : str,
        bias: Optional[torch.Tensor] = None
    ) -> None:
    assert d.shape[-2:] == (m, n), f"d shape mismatch: {d.shape} vs ({m}, {n})"
    return

def blaslt_w8a8_bf16_gemm_wrapper(a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        d : torch.Tensor,
        m : int,
        n : int,
        k : int,
        batch : int,
        transpose_flag : str,
        bias: Optional[torch.Tensor] = None
    ) -> None:
    alpha = torch.tensor(1, dtype=torch.int32, device='cpu')
    beta = torch.tensor(0, dtype=torch.int32, device='cpu')
    lmslimquant.blaslt_w8a8_bf16_gemm(a, b, scale_a, scale_b, d, m, n, k, batch, transpose_flag, alpha, beta, bias)
    return

def blaslt_w8a8_bf16_gemm_fake(a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        d : torch.Tensor,
        m : int,
        n : int,
        k : int,
        batch : int,
        transpose_flag : str,
        bias: Optional[torch.Tensor] = None
    ) -> None:
    assert d.shape[-2:] == (m, n), f"d shape mismatch: {d.shape} vs ({m}, {n})"
    return

direct_register_custom_op(
    op_name = "blaslt_w8a8_fp16_gemm",
    op_func = blaslt_w8a8_fp16_gemm_wrapper,
    mutates_args = ["d"],  # in-place
    target_lib = lmslim_lib,
    fake_impl = blaslt_w8a8_fp16_gemm_fake,
)

direct_register_custom_op(
    op_name = "blaslt_w8a8_bf16_gemm",
    op_func = blaslt_w8a8_bf16_gemm_wrapper,
    mutates_args = ["d"],  # in-place
    target_lib = lmslim_lib,
    fake_impl = blaslt_w8a8_bf16_gemm_fake,
)

def hipblaslt_w8a8_gemm(a : torch.Tensor,
        b : torch.Tensor,
        scale_a : torch.Tensor,
        scale_b : torch.Tensor,
        m : int,
        n : int,
        k : int,
        transpose_flag : str,
        out_dtype : torch.dtype,
        bias : Optional[torch.Tensor] = None # reserve
    ) -> tuple[bool, torch.Tensor]:
    assert transpose_flag == 'NT'
    assert a.is_contiguous() and b.is_contiguous()
    assert scale_a.dtype is torch.float32 and scale_b.dtype is torch.float32
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.dtype == out_dtype
    assert a.shape[-1] == b.shape[-1]

    d = torch.empty((m, n), dtype=out_dtype, device = 'cuda')

    if out_dtype == torch.float16:
        torch.ops.lmslim.blaslt_w8a8_fp16_gemm(
                b, a, scale_b, scale_a, d, m, n, k, b.shape[0] if b.dim() == 3 else 1, 'TN', bias)
        return True, d
    elif out_dtype == torch.bfloat16:
        torch.ops.lmslim.blaslt_w8a8_bf16_gemm(
                b, a, scale_b, scale_a, d, m, n, k, b.shape[0] if b.dim() == 3 else 1, 'TN', bias)
        return True, d
    else:
        raise ValueError(f"Unsupported out_dtype: {out_dtype}")

def lo_rms_norm_dynamic_per_token_quant_real(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    residual: Optional[torch.Tensor] = None,
    update_input: Optional[bool] = True,
    limit: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, device=input.device, dtype=quant_dtype)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                        device=input.device,
                        dtype=torch.float32)
    rms_norm_dynamic_per_token_quant(output, input, weight,
                                               scales, epsilon,
                                               residual, update_input, limit)
    return output, scales

def lo_rms_norm_dynamic_per_token_quant_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    residual: Optional[torch.Tensor] = None,
    update_input: Optional[bool] = True,
    limit: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile"""
    output = torch.zeros_like(input, dtype=quant_dtype)
    scales = torch.ones((input.numel() // input.shape[-1], 1),
                        device=input.device,
                        dtype=torch.float32)
    return output, scales

direct_register_custom_op(
    op_name="lo_rms_norm_dynamic_per_token_quant",
    op_func=lo_rms_norm_dynamic_per_token_quant_real,
    mutates_args=[],
    target_lib=lmslim_lib,
    fake_impl=lo_rms_norm_dynamic_per_token_quant_fake,
)

def lm_faster_rmsquant(input: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    residual: Optional[torch.Tensor] = None,
    update_input: Optional[bool] = True,
    limit: float = -1.0):
    i_q, _scales = torch.ops.lmslim.lo_rms_norm_dynamic_per_token_quant(input=input,
                                                                        weight=rms_weight,
                                                                        epsilon=epsilon,
                                                                        quant_dtype=quant_dtype,
                                                                        residual=residual,
                                                                        update_input=update_input,
                                                                        limit=limit)
    return i_q, _scales

def lo_lm_fuse_silu_mul_quant_real(input: torch.Tensor
                                   ) -> tuple[torch.Tensor, torch.Tensor]:
    return fuse_silu_mul_quant(input)

def lo_lm_fuse_silu_mul_quant_fake(input: torch.Tensor
                                   ) -> tuple[torch.Tensor, torch.Tensor]:
    leading, d2 = input.shape
    d = d2 // 2
    output = torch.empty(leading, d, dtype=torch.int8, device=input.device)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                        device=input.device,
                        dtype=torch.float32)
    return output, scales

direct_register_custom_op(
    op_name="lo_lm_fuse_silu_mul_quant",
    op_func=lo_lm_fuse_silu_mul_quant_real,
    mutates_args=[],
    target_lib=lmslim_lib,
    fake_impl=lo_lm_fuse_silu_mul_quant_fake,
)

def lm_fuse_silu_mul_quant(input: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    # return fuse_silu_mul_quant(input)
    return torch.ops.lmslim.lo_lm_fuse_silu_mul_quant(input)
