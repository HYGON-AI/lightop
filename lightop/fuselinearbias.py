import math
import os

from torch.autograd import Function
import torch
from lightop import op
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init
# import numpy as np

os.environ['ROCBLAS_TENSILE_GBF_LIBPATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'librarygemmbiasf')


class LinearBiasFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, grad_weight, grad_bias):
        ctx.save_for_backward(input, weight, bias, grad_weight, grad_bias)
        # if input.size()[0] == 107 and weight.size()[1] == 1024 and weight.size()[0] == 1024 and bias is not None:
        #     output = op.linearbias_forward(bias, input, weight.t())
        #     #print("fuse forward!!!!!!!!!!!!!!!!!!!!!1")
        # else:
        #     output = input.mm(weight.t())
        #     if bias is not None:
        #         output += bias.unsqueeze(0).expand_as(output)
        #     #print("nofuse forward!!!!!!!!!!!!!!!!!!!!!1")
        output = op.linearbias_forward(bias, input, weight.t())

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, grad_weight, grad_bias = ctx.saved_tensors
 
        grad_input = None

        if ctx.needs_input_grad[0]:
            if grad_output.dim() == 2:
                grad_input = grad_output.mm(weight)
            else:
                grad_input = grad_output.matmul(weight)
       
        #grad_input = grad_weight = grad_bias = None
        #
        #if ctx.needs_input_grad[0]:
        #    grad_input = grad_output.mm(weight)
        #if ctx.needs_input_grad[1]:
        #    grad_weight = grad_output.t().mm(input)
        #if bias is not None and ctx.needs_input_grad[2]:
        #    grad_bias = grad_output.sum(0)
 
        if input.size()[0] == 107 and weight.size()[1] == 1024 and weight.size()[0] == 1024 and bias is not None:
            if ctx.needs_input_grad[1] and ctx.needs_input_grad[2]:
                #grad_bias = torch.empty_like(bias, memory_format=torch.contiguous_format)
                #grad_bias = torch.empty(bias.size(), dtype=bias.dtype, layout=bias.layout, device=bias.device)
                if grad_output.dim() == 2:
                    op.linearbias_backward(grad_bias, grad_weight, grad_output.t(), input)
                else:
                    op.linearbias_backward(grad_bias, grad_weight, grad_output.view(-1, grad_output.shape[-1]).t(), input.view(-1, input.shape[-1]))
                #grad_weight = op.linearbias_backward(grad_bias, input, grad_output)
                #grad_weight = grad_output.t().mm(input)
                #grad_weight = input.t().mm(grad_output)
                #grad_bias = grad_output.sum(0
            #print("fuse backward!!!!!!!!!!!!!!!!!!!!!1")
        else:
            grad_weight = grad_bias = None
            if ctx.needs_input_grad[1]:
                if grad_output.dim() == 2:
                    grad_weight = grad_output.t().mm(input)
                else:
                    grad_weight = grad_output.view(-1, grad_output.shape[-1]).t().matmul(input.view(-1, input.shape[-1]))
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0)
            #print("nofuse backward!!!!!!!!!!!!!!!!!!!!!1")

        return grad_input, grad_weight, grad_bias, None, None


class LinearBias(torch.nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    bias: Tensor
    grad_weight: Tensor
    grad_bias: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearBias, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        #self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        if self.weight.requires_grad == True:
            self.grad_weight = torch.empty(self.weight.size(), dtype=self.weight.dtype, layout=self.weight.layout, device=self.weight.device)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            if self.bias.requires_grad == True:
                self.grad_bias = torch.empty(self.bias.size(), dtype=self.bias.dtype, layout=self.bias.layout, device=self.bias.device)
        else:
            self.register_parameter('bias', None)
            self.register_parameter('grad_bias', None)
        self.reset_parameters()
        op.create_gemmbias_handle()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    def forward(self, input: Tensor) -> Tensor:
        return LinearBiasFunction.apply(input, self.weight, self.bias, self.grad_weight, self.grad_bias)

    def __del__(self):
        op.destroy_gemmbias_handle()