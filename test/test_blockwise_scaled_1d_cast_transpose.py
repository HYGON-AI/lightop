

import math
import time
import os
import torch
import dataclasses
from typing import Optional, Protocol, Tuple,Iterable, Union
from torch.utils.cpp_extension import IS_HIP_EXTENSION
from lightop import blockwise_scaled_1d_cast_transpose

blockwise_fp8_block_len=128


def scale_from_amax_tensor(
    x_dtype: torch.dtype,
    amax: torch.Tensor,
    quant_dtype: torch.dtype,
    *,
    eps: float,
    pow_2_scales: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derives quantization and dequantization from amax and options.

    Reference implementation for scale calculation.

    Returns:
    - scale: quantization scales
    - scale_inv: dequantization scales
    - amax: Amax tensor with updates made for extrema values.
    """
    assert amax.dtype == torch.float, "amax must be a float tensor."
    if quant_dtype == torch.int8:
        fp8_max = torch.iinfo(quant_dtype).max
    else:
        fp8_max = torch.finfo(quant_dtype).max
    # Clamping amax to avoid division by small numbers
    amax = torch.max(amax, torch.tensor(eps))

    # Compute scale factor
    scale = torch.div(fp8_max, amax)
    # Note frexp doesn't give back inf for exponent with an inf input
    # We take care of inf before pow_2_scales
    scale = torch.where(scale == torch.inf, torch.finfo(x_dtype).max, scale)
    if pow_2_scales:
        # Calculate rounded down exponent
        _, exp = torch.frexp(scale)
        # Positive numbers are always returned as mant, exp with
        # a mantissa in [0.5, 1.0). Because a normal float has a mantissa with
        # hidden bit in [1.0, 2.0), the exponent will be off by exactly one because
        # of the shift. Subnormal and zero cases need not be considered because
        # the smallest possible result of fp8_max / amax is still normal.
        exp = exp - 1
        # No subnormals and zero.
        assert (exp > -127).all()
        unity = torch.tensor([1.0], device=exp.device)
        if IS_HIP_EXTENSION:
            host_scale = torch.ldexp(unity.cpu(), exp.cpu())
            scale = host_scale.to(exp.device)
        else:
            torch.ldexp(unity, exp, out=scale)
        # Case where amax is inf. The frexp, ldexp logic changes 0.0 scales
        # Return 0.0 for 0.0 scale for consistency with non-pow2 scale
        # calculation.
        scale = torch.where(amax == float("inf"), 0.0, scale)

    # Handle overflow cases for amax zero causing NaN
    scale = torch.where(amax == 0, 1.0, scale)

    # Compute scale_inv
    scale_inv = torch.reciprocal(scale)

    return scale, scale_inv, amax

@dataclasses.dataclass()
class QuantizeResult:
    data: torch.Tensor
    scale: torch.Tensor
    data_t: Optional[torch.Tensor]
    scale_t: Optional[torch.Tensor]


@dataclasses.dataclass()
class CuBLASScaleMunger:
    def munge_scale_shapes_for_backend(
        self,
        unmunged: QuantizeResult,
        tile_shape: Tuple[int, int],
    ) -> QuantizeResult:
        """
        cuBLAS GEMMs requires 1x128 quantized tensors to be have scales transposed
        so that for an (M, N) tensor, the scales are (RoundUpDiv(N, 128), RoundUp(M, 4))

        For 128x128 quantized tensors, the GEMM expects (M, PadToAlign(RoundUpDivide(N, 128), 4))
        format. If RoundUpDivide(N, 128) is not divisible by 4, a transformation is required
        """

        def _pad_inner_to_align(s: torch.Tensor, transpose: bool) -> torch.Tensor:
            if transpose:
                s = s.transpose(-1, -2).contiguous()
            M, K = s.shape
            if K % 4 == 0:
                return s
            k_pad = 4 - (K % 4)
            return torch.nn.functional.pad(s, (0, k_pad), mode="constant", value=0).contiguous()

        s = _pad_inner_to_align(unmunged.scale, transpose=tile_shape[0] == 1)
        if unmunged.scale_t is None:
            s_t = None
        else:
            s_t = _pad_inner_to_align(unmunged.scale_t, transpose=tile_shape[0] == 1)
        return QuantizeResult(unmunged.data, s, unmunged.data_t, s_t)

    def demunge_scale_shape_from_backend(
        cls,
        qtensor_shape: Tuple[int, int],
        scales: torch.Tensor,
        tile_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Inverse operation of munge_scale_shapes_for_backend
        """
        if tile_shape[0] != 1:
            # 2D block quantized tensor may need padding stripped off
            derived_scale_k_shape = math.ceil(qtensor_shape[1] / tile_shape[1])
        else:
            derived_scale_k_shape = qtensor_shape[0]
        M, K = scales.shape
        if derived_scale_k_shape != K:
            scales = scales[:, :derived_scale_k_shape].contiguous()
        if tile_shape[0] == 1:
            return scales.transpose(-1, -2).contiguous()
        else:
            return scales


@dataclasses.dataclass()
class BlockwiseQuantizerReference:
    """
    A reference QuantizeOp for subchannel/block hybrid quantization.

    Defers to ref GEMMs and quantizization formatting based on the backend.
    """

    def __init__(self) -> None:
        self.scale_munger = CuBLASScaleMunger()

 
    def _quantize_square_block_tiling(
        cls,
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        tile_len: int,
        *,
        return_transpose: bool,
        pow_2_scales: bool,
        eps: float,
    ) -> QuantizeResult:
        M, K = x.shape

        pad_m_k = [0, 0]
        if K % tile_len != 0:
            pad_m_k[1] = tile_len - (K % tile_len)
        if M % tile_len != 0:
            pad_m_k[0] = tile_len - (M % tile_len)

        unpadded_m, unpadded_k = M, K
        if pad_m_k[0] != 0 or pad_m_k[1] != 0:
            x = torch.nn.functional.pad(
                x, (0, pad_m_k[1], 0, pad_m_k[0]), mode="constant", value=0
            ).contiguous()
            M, K = x.shape

        x_tiled = x.reshape(M // tile_len, tile_len, K // tile_len, tile_len)
        amax_grid = (
            torch.abs(x_tiled.transpose(-3, -2))
            .reshape(M // tile_len, K // tile_len, tile_len**2)
            .amax(dim=-1)
        ).float()
        if quant_dtype == torch.int8:
            dtype_max = torch.iinfo(quant_dtype).max
        else:
            dtype_max = torch.finfo(quant_dtype).max

        scale, scale_inv, _ = scale_from_amax_tensor(
            x_dtype=x.dtype,
            amax=amax_grid,
            quant_dtype=quant_dtype,
            pow_2_scales=pow_2_scales,
            eps=eps,
        )
        qx = x_tiled * scale.reshape(M // tile_len, 1, K // tile_len, 1)
        qx = torch.clamp(qx, min=-dtype_max, max=dtype_max)
        if quant_dtype == torch.int8:
            positive_mask = qx >= 0
            negative_mask = ~positive_mask
            pos_part = torch.where(positive_mask, torch.floor(qx + 0.5), 0)
            neg_part = torch.where(negative_mask, torch.ceil(qx - 0.5), 0)
            qx = pos_part + neg_part
        qx = qx.to(dtype=quant_dtype)
        qx = qx.reshape(M, K)
        if unpadded_k != K or unpadded_m != M:
            qx = qx[:unpadded_m, :unpadded_k].contiguous()
        if return_transpose:
            # Valid because of square block sizes
            qx_t = qx.transpose(-1, -2).contiguous()
            scale_inv_t = scale_inv.transpose(-1, -2).contiguous()
        else:
            qx_t = None
            scale_inv_t = None

        return QuantizeResult(data=qx, scale=scale_inv, data_t=qx_t, scale_t=scale_inv_t)


    def _quantize_vectorwise_reference(
        cls,
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        tile_len: int,
        *,
        pow_2_scales: bool,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        M, K = x.shape
        if quant_dtype == torch.int8:
            dtype_max = torch.iinfo(quant_dtype).max
        else:
            dtype_max = torch.finfo(quant_dtype).max
        x_tiled = x.reshape(M, K // tile_len, tile_len)
        amax_grid = torch.abs(x_tiled).amax(dim=-1).float()
        scale, scale_inv, _ = scale_from_amax_tensor(
            x_dtype=x.dtype,
            amax=amax_grid,
            quant_dtype=quant_dtype,
            pow_2_scales=pow_2_scales,
            eps=eps,
        )
        qx = x_tiled * scale.reshape(M, K // tile_len, 1)
        qx = torch.clamp(qx, min=-dtype_max, max=dtype_max)
        if quant_dtype == torch.int8:
            positive_mask = qx >= 0
            negative_mask = ~positive_mask
            pos_part = torch.where(positive_mask, torch.floor(qx + 0.5), 0)
            neg_part = torch.where(negative_mask, torch.ceil(qx - 0.5), 0)
            qx = pos_part + neg_part
        qx = qx.to(dtype=quant_dtype)
        qx = qx.reshape(M, K)
        return qx, scale_inv


    def _quantize_vector_tiling(
        cls,
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        tile_len: int,
        *,
        return_transpose: bool,
        pow_2_scales: bool,
        eps: float,
    ) -> QuantizeResult:
        M, K = x.shape

        if K % tile_len == 0:
            qref_input = x
        else:
            pad_amount = tile_len - (K % tile_len)
            pad = (0, pad_amount)
            qref_input = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        qout_padded, scale_inv = cls._quantize_vectorwise_reference(
            qref_input,
            quant_dtype,
            tile_len=tile_len,
            pow_2_scales=pow_2_scales,
            eps=eps,
        )
        if K % tile_len == 0:
            qout = qout_padded
        else:
            qout = qout_padded[:, :K].contiguous()

        if return_transpose:
            if M % tile_len == 0:
                qref_input = x.transpose(-1, -2).contiguous()
            else:
                amount_to_pad = tile_len - (M % tile_len)
                pad = (0, amount_to_pad)
                qref_input = torch.nn.functional.pad(
                    x.transpose(-1, -2), pad, mode="constant", value=0
                ).contiguous()
            qout_t_padded, scale_inv_t = cls._quantize_vectorwise_reference(
                qref_input,
                quant_dtype,
                tile_len=tile_len,
                pow_2_scales=pow_2_scales,
                eps=eps,
            )
            if M % tile_len == 0:
                qout_t = qout_t_padded
            else:
                qout_t = qout_t_padded[:, :M].contiguous()
        else:
            qout_t, scale_inv_t = None, None

        return QuantizeResult(data=qout, scale=scale_inv, data_t=qout_t, scale_t=scale_inv_t)

    def quantize(
        self,
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        return_transpose: bool = False,
        eps: float = 0.0,
        pow_2_scales: bool = False,
        quant_tile_shape: Tuple[int, int] = (blockwise_fp8_block_len, blockwise_fp8_block_len),
        munge_scale_shapes: bool = True,
    ) -> QuantizeResult:
        # sanity checks
        assert x.dim() == 2
        assert x.dtype in (
            torch.float,
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ), "Unsupported input dtype."
        assert quant_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.int8,
        ), "Unsupported quant dtype."

        assert quant_tile_shape in ((1, blockwise_fp8_block_len), (blockwise_fp8_block_len, blockwise_fp8_block_len))
        if quant_tile_shape[0] == 1:
            # Quantize row-wise
            result = self._quantize_vector_tiling(
                x,
                quant_dtype,
                tile_len=quant_tile_shape[1],
                return_transpose=return_transpose,
                pow_2_scales=pow_2_scales,
                eps=eps,
            )
            if munge_scale_shapes:
                result = self.scale_munger.munge_scale_shapes_for_backend(
                    result,
                    quant_tile_shape,
                )
            return result
        else:
            # Quantize block-wise
            result = self._quantize_square_block_tiling(
                x,
                quant_dtype,
                tile_len=quant_tile_shape[0],
                return_transpose=return_transpose,
                pow_2_scales=pow_2_scales,
                eps=eps,
            )
            if munge_scale_shapes:
                result = self.scale_munger.munge_scale_shapes_for_backend(
                    result,
                    quant_tile_shape,
                )
            return result


def initialize_for_many_scales(
    x_shape_2d: Tuple[int, int], tile_shape: Tuple[int, int], *, dtype: torch.dtype, device: str
) -> torch.Tensor:
    """
    Put separate distributions into each quantization tile
    to avoid many tiles having similar scale values and
    causing false passes.
    """
    tile_grid_shape = (
        math.ceil(x_shape_2d[0] / tile_shape[0]),
        math.ceil(x_shape_2d[1] / tile_shape[1]),
    )
    # Arbitrary size
    max_val = 8192.0
    # Make a uniform distribution of [-max_val, max_val]
    tile_extrema = torch.rand(*tile_grid_shape, dtype=dtype) * max_val * 2 - max_val
    result = torch.empty(x_shape_2d, dtype=dtype, device=device)
    tile_elements = tile_shape[0] * tile_shape[1]
    for i in range(tile_grid_shape[0]):
        for j in range(tile_grid_shape[1]):
            target = tile_extrema[i, j].item()
            step = target / (tile_elements)
            if target == 0:
                tile = torch.zeros(tile_shape, dtype=dtype, device=device)
            else:
                tile = torch.arange(0.0, target, step=step, dtype=dtype, device=device)
            tile = tile.reshape(*tile_shape)
            min_dst_vals = (i * tile_shape[0], j * tile_shape[1])
            max_dst_vals = (
                min((i + 1) * tile_shape[0], x_shape_2d[0]),
                min((j + 1) * tile_shape[1], x_shape_2d[1]),
            )
            max_src_vals = (
                max_dst_vals[0] - min_dst_vals[0],
                max_dst_vals[1] - min_dst_vals[1],
            )
            result[min_dst_vals[0] : max_dst_vals[0], min_dst_vals[1] : max_dst_vals[1]] = tile[
                : max_src_vals[0], : max_src_vals[1]
            ]
    return result

def round_up_to_nearest_multiple(value, multiple):
    """Round up `value` to the next mutiple of `multiple`"""
    if multiple == 0:
        raise ValueError("multiple cannot be zero.")
    return ((value + multiple - 1) // multiple) * multiple

def get_scale_shape(shape: Iterable[int], columnwise: bool,all_gather_usage: bool) -> Tuple[int, int]:
    
    M, K = 1, 1
    for i in range(len(shape) - 1):
        M *= shape[i]
    if len(shape) > 0:
        K = shape[-1]

    if columnwise:
        columnwise_compact =all_gather_usage
        outer = math.ceil(M / blockwise_fp8_block_len)
        inner = round_up_to_nearest_multiple(K, 4) if not columnwise_compact else K
       
        return (outer, inner)
    # rowwise
    rowwise_compact = all_gather_usage
    outer = math.ceil(K / blockwise_fp8_block_len)
    inner = round_up_to_nearest_multiple(M, 4) if not rowwise_compact else M
   
    return (outer, inner) if not rowwise_compact else (inner, outer)


def get_columnwise_shape(shape: Iterable[int]) -> Tuple[int, ...]:
        
        if len(shape) == 0:
            return tuple()

        colwise_shape = [shape[-1]]
        for i in range(len(shape) - 1):
            colwise_shape.append(shape[i])
        return tuple(colwise_shape)


def test_quantization_1D_block_tiling_with_compact_data_and_scales(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quant_dtype: torch.dtype,
    eps: float,
    pow_2_scales: bool,
) -> None:
   
   tile_size = (1, blockwise_fp8_block_len)
   ref_quantizer = BlockwiseQuantizerReference()
   # Setup device and random seed
   device = "cuda"
   seed = 0
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   
   # Input
   x = initialize_for_many_scales((M, N), tile_size, dtype=x_dtype, device=device)
   scale_inv_shape = get_scale_shape((M, N), columnwise=False,all_gather_usage=True)
   rowwise_data=torch.empty((M, N), dtype=torch.uint8, device=device)
   rowwise_scale_inv=torch.empty(scale_inv_shape,dtype=torch.float32,device=device)
 
   columnwise_data=torch.empty((M, N), dtype=torch.uint8, device=device)
   columnwise_scale_shape =get_scale_shape((M, N), columnwise=True,all_gather_usage=True)
   columnwise_scale_inv=torch.empty(columnwise_scale_shape,dtype=torch.float32,device=device)


   for i in range(10):
        blockwise_scaled_1d_cast_transpose(x,rowwise_data,rowwise_scale_inv, columnwise_data,columnwise_scale_inv, 
                                           None,eps,pow_2_scales,True,quant_dtype)

   torch.cuda.synchronize()

   start=torch.cuda.Event(enable_timing=True)
   end=torch.cuda.Event(enable_timing=True)
   start.record()
   for i in range(100):
        
        blockwise_scaled_1d_cast_transpose(x,rowwise_data,rowwise_scale_inv, columnwise_data,columnwise_scale_inv, 
                                           None,eps,pow_2_scales,True,quant_dtype)

   end.record()
   torch.cuda.synchronize() 
   fc1_espl= start.elapsed_time(end)
   print('blockwise_scaling_exact_compact', 'averange time is', fc1_espl/ 100, 'ms') 

   assert rowwise_data is not None
   qx: torch.Tensor = rowwise_data.view(dtype=quant_dtype)
   assert rowwise_scale_inv is not None
   sx: torch.Tensor = rowwise_scale_inv
   qx_t = columnwise_data
   sx_t = columnwise_scale_inv


   qresult_ref = ref_quantizer.quantize(
        x,
        quant_dtype=quant_dtype,
        return_transpose=True,
        eps=eps,
        pow_2_scales=pow_2_scales,
        quant_tile_shape=(1, blockwise_fp8_block_len),
        munge_scale_shapes=False,
    )
  
   qx_ref, sx_ref, qx_t_ref, sx_t_ref = (
        qresult_ref.data,
        qresult_ref.scale,
        qresult_ref.data_t,
        qresult_ref.scale_t,
    )

    # match the reference quantize transpose output with the columnwise non-transpose method
   qx_t_ref = qx_t_ref.transpose(-1, -2).contiguous()
   sx_t_ref = sx_t_ref.transpose(-1, -2).contiguous()
   # Check
   torch.testing.assert_close(qx.float(), qx_ref.float(), atol=0.0, rtol=0.0)
   torch.testing.assert_close(sx, sx_ref, atol=0.0, rtol=0.0)
   assert qx_t is not None
   qx_t = qx_t.view(dtype=quant_dtype)
   assert qx_t_ref is not None
   assert sx_t is not None
   assert sx_t_ref is not None

   torch.testing.assert_close(qx_t.float(), qx_t_ref.float(), atol=0.0, rtol=0.0)
   torch.testing.assert_close(sx_t, sx_t_ref, atol=0.0, rtol=0.0)


def test_quantization_1D_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quant_dtype: torch.dtype,
    eps: float,
    return_transpose: bool,
    pow_2_scales: bool,
) -> None:
   
   tile_size = (1, blockwise_fp8_block_len)
   ref_quantizer = BlockwiseQuantizerReference()
   # Setup device and random seed
   device = "cuda"
   seed = 0
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   
   # Input
   x = initialize_for_many_scales((M, N), tile_size, dtype=x_dtype, device=device)
   scale_inv_shape = get_scale_shape((M, N), columnwise=False,all_gather_usage=False)
   rowwise_data=torch.empty((M, N), dtype=torch.uint8, device=device)
   rowwise_scale_inv=torch.empty(scale_inv_shape,dtype=torch.float32,device=device)
 
   if return_transpose:
        columnwise_data=torch.empty(get_columnwise_shape((M, N)), dtype=torch.uint8, device=device)
        columnwise_scale_shape =get_scale_shape((M, N), columnwise=True,all_gather_usage=False)
        columnwise_scale_inv=torch.empty(columnwise_scale_shape,dtype=torch.float32,device=device)
   else:
        columnwise_data=None
        columnwise_scale_inv=None
 
   for i in range(10):
        blockwise_scaled_1d_cast_transpose(x,rowwise_data,rowwise_scale_inv, columnwise_data,columnwise_scale_inv, 
                                           None,eps,pow_2_scales,False,quant_dtype)

   torch.cuda.synchronize()

   start=torch.cuda.Event(enable_timing=True)
   end=torch.cuda.Event(enable_timing=True)
   start.record()
   for i in range(100):
        
        blockwise_scaled_1d_cast_transpose(x,rowwise_data,rowwise_scale_inv, columnwise_data,columnwise_scale_inv, 
                                           None,eps,pow_2_scales,False,quant_dtype)

   end.record()
   torch.cuda.synchronize() 
   fc1_espl= start.elapsed_time(end)
   print('blockwise_scaling_exact', 'averange time is', fc1_espl/ 100, 'ms')

   assert rowwise_data is not None
   qx: torch.Tensor = rowwise_data.view(dtype=quant_dtype)
   assert rowwise_scale_inv is not None
   sx: torch.Tensor = rowwise_scale_inv
   qx_t = columnwise_data
   sx_t = columnwise_scale_inv


   qresult_ref = ref_quantizer.quantize(
        x,
        quant_dtype=quant_dtype,
        return_transpose=return_transpose,
        eps=eps,
        pow_2_scales=pow_2_scales,
        quant_tile_shape=(1, blockwise_fp8_block_len),
    )
  
   qx_ref, sx_ref, qx_t_ref, sx_t_ref = (
        qresult_ref.data,
        qresult_ref.scale,
        qresult_ref.data_t,
        qresult_ref.scale_t,
    )

   # Check
   torch.testing.assert_close(qx.float(), qx_ref.float(), atol=0.0, rtol=0.0)
   scale_mask = torch.ones(
        (math.ceil(M / tile_size[0]), math.ceil(N / tile_size[1])), device=sx.device )
   scale_mask = ref_quantizer.scale_munger.munge_scale_shapes_for_backend(
        QuantizeResult(qx, scale_mask, None, None), tile_size
    ).scale
   sx = sx * scale_mask
   torch.testing.assert_close(sx, sx_ref, atol=0.0, rtol=0.0)
   
   if return_transpose:
        assert qx_t is not None
        qx_t = qx_t.view(dtype=quant_dtype)
        assert qx_t_ref is not None
        assert sx_t is not None
        assert sx_t_ref is not None
        scale_mask = torch.ones(
        (math.ceil(N / tile_size[0]), math.ceil(M / tile_size[1])),
        device=sx_t.device,)
        scale_mask = ref_quantizer.scale_munger.munge_scale_shapes_for_backend(
        QuantizeResult(qx_t, scale_mask, None, None), tile_size ).scale
        # print(sx_t,sx_t_ref)
        sx_t = sx_t * scale_mask

        torch.testing.assert_close(qx_t.float(), qx_t_ref.float(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(sx_t, sx_t_ref, atol=0.0, rtol=0.0)
   else:
        # should be None
        assert qx_t is None and qx_t_ref is None
        assert sx_t is None and sx_t_ref is None

if __name__ == "__main__":

    m=2048
    n=2000
    x_dtype=torch.float32
    quant_dtype=torch.int8
    eps=0
    pow_2_scales=True
    compact=True
    return_transpose=True

    if compact==True:
    
        test_quantization_1D_block_tiling_with_compact_data_and_scales(x_dtype,m,n,quant_dtype,eps,pow_2_scales)
    
    else :  
        test_quantization_1D_block_tiling_versus_reference(x_dtype,m,n,quant_dtype,eps,return_transpose,pow_2_scales)

   
       
       