from typing import Optional

import torch
from torch import Tensor
from torch.autograd import Function
from .fusebnrelu import _FuseBaseModule
from .fuseactmode import fuseactmode
from lightop import op

class _FuseBatchNormAct(Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight, bias, running_mean, running_var, training, momentum, eps, track_running_stats, num_batches_tracked, actmode=fuseactmode.relu.value) -> Tensor:
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

        forward_data = op.bnrelu_forward(
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
            int(actmode)
        )
        #grad_out = forward_data[0]
        #save_mean = forward_data[1]
        #save_invstd = forward_data[2]
        ctx.eps = eps
        ctx.bn_training = bn_training
        ctx.impl_index = forward_data[4]
        ctx.actmode = actmode
        ctx.save_for_backward(input, forward_data[0], forward_data[1], weight, bias, forward_data[2], forward_data[3], running_mean, running_var)
        return forward_data[0]
    
    @staticmethod
    def backward(ctx, gradOutput) -> Tensor:
        save_input, save_output_relu, save_relu_in, weight, bias, save_mean, save_invstd, running_mean, running_var = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = op.bnrelu_backward(
            ctx.impl_index,
            gradOutput, 
            save_input,
            save_output_relu,
            save_relu_in,
            weight,
            bias,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            ctx.bn_training,
            ctx.eps,
            ctx.needs_input_grad[0:3],
            int(ctx.actmode)
            )
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None

class FuseBatchNormAct2d(_FuseBaseModule):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return _FuseBatchNormAct.apply(input, self.weight, self.bias, self.running_mean, self.running_var,
                                        self.training, self.momentum, self.eps,
                                        self.track_running_stats, self.num_batches_tracked, self.actmode)
