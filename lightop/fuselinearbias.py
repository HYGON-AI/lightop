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
os.environ['ROCBLAS_TENSILE_GDBB_LIBPATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'librarygemmdbiasb')
libpath=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib/')
assert_line='wrong envirenment! please execute \"export LD_LIBRARY_PATH='+libpath +':$LD_LIBRARY_PATH\" and write this line to ~/.bashrc'
assert libpath in os.environ['LD_LIBRARY_PATH'] , assert_line

class LinearBiasFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, grad_weight, grad_bias):
        ctx.save_for_backward(input, weight, bias, grad_weight, grad_bias)
        # if input.dim() == 2:
        #     output = torch.addmm(bias, input, weight.t())
        # else:
        #     output = torch.matmul(input, weight.t())
        #     output += bias.unsqueeze(0).expand_as(output)

        output = op.linearbias_forward(bias, input, weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, grad_weight, grad_bias = ctx.saved_tensors
 
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            if grad_output.dim() == 2:
                grad_input = grad_output.mm(weight)
            else:
                grad_input = grad_output.matmul(weight)
        
        if ctx.needs_input_grad[1] and ctx.needs_input_grad[2]:
            if grad_output.dim() == 2:
                grad_weight = grad_output.t().mm(input)
            else:
                grad_weight = grad_output.flatten(0,-2).t().mm(input.flatten(0,-2))
            grad_bias = grad_output.sum(0)

        # if ctx.needs_input_grad[1] and ctx.needs_input_grad[2]:
        #     if grad_output.dim() == 2:
        #         op.linearbias_backward(grad_bias, grad_weight, grad_output.t(), input)
        #     else:
        #         op.linearbias_backward(grad_bias, grad_weight, grad_output.flatten(0,-2).t(), input.flatten(0,-2))

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
        # self.weight = Parameter(torch.ones((out_features, in_features), **factory_kwargs))
        # if self.weight.requires_grad == True:
        #     self.grad_weight = torch.empty(self.weight.size(), dtype=self.weight.dtype, layout=self.weight.layout, device=self.weight.device)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            # self.bias = Parameter(torch.ones((2,out_features), **factory_kwargs))
            # if self.bias.requires_grad == True:
            #     self.grad_bias = torch.empty(self.bias.size(), dtype=self.bias.dtype, layout=self.bias.layout, device=self.bias.device)
        else:
            self.register_parameter('bias', None)
            # self.register_parameter('grad_bias', None)
        self.reset_parameters()
        op.create_gemmbias_handle()
        op.create_gemmdbias_handle()


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
        # return LinearBiasFunction.apply(input, self.weight, self.bias, self.grad_weight, self.grad_bias)
        # return op.fused_linear_bias_forward_autograd(input, self.weight, self.bias, self.grad_weight, self.grad_bias)
        return op.fused_linear_bias_forward_autograd(input, self.weight, self.bias)

    def __del__(self):
        op.destroy_gemmbias_handle()
        op.destroy_gemmdbias_handle()