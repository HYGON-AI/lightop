from typing import Optional

import torch
from torch import Tensor
from torch.nn.parameter import Parameter 
from torch.autograd import Function
from .fusebnrelu import _FuseBaseModule
from torch.nn import init
from lightop import op

class _FuseBatchNormDropout(Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight, bias, running_mean, running_var, training, momentum, eps, track_running_stats, num_batches_tracked, p = float(0.5)) -> Tensor:
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = momentum

        if training and track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if num_batches_tracked is not None:  # type: ignore[has-type]
                num_batches_tracked = num_batches_tracked + 1  # type: ignore[has-type]
                if momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = momentum

        if training:
            bn_training = True
        else:
            bn_training = (running_mean is None) and (running_var is None)

        forward_data = op.bn_dropout_forward(
            input,
            weight,
            bias,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean
            if not training or track_running_stats
            else None,
            running_var if not training or track_running_stats else None,
            bn_training,
            exponential_average_factor,
            eps,
            p
        )
        ctx.eps = eps
        ctx.bn_training = bn_training
        ctx.p = p
        ctx.save_for_backward(input, forward_data[1], weight, bias, forward_data[2], forward_data[3], running_mean, running_var)
        return forward_data[0]
    
    @staticmethod
    def backward(ctx, gradOutput) -> Tensor:
        save_input, save_output_dropout, weight, bias, save_mean, save_invstd, running_mean, running_var = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = op.bn_dropout_backward(
            gradOutput, 
            save_input,
            save_output_dropout,
            weight,
            bias,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            ctx.bn_training,
            ctx.eps,
            ctx.needs_input_grad[0:3],
            ctx.p
            )
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None


    

class FuseBatchNormDropout2d(_FuseBaseModule):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
        if (self.p < 0 or self.p > 1):
            raise ValueError("p expected [0,1] (got )".format(self.p))

    def forward(self, input):
        self._check_input_dim(input)
        return _FuseBatchNormDropout.apply(input, self.weight, self.bias, self.running_mean, self.running_var,
                                        self.training, self.momentum, self.eps,
                                        self.track_running_stats, self.num_batches_tracked,self.p)
