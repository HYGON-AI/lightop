from torch.autograd import Function
import torch
from torch import Tensor
from . import op

def destindex_copy_quantize_int8_cuda(input, scale_zero, output):
    op.destindex_copy_quantize_int8_cuda(input, scale_zero, output)

def destindex_dequantize_int8_cuda(input, scale_zero, output):
    op.destindex_dequantize_int8_cuda(input, scale_zero, output)
