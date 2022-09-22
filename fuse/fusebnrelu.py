from typing import Optional

import torch
from torch import Tensor
from torch.nn.parameter import Parameter 
from torch.autograd import Function

from torch.nn import init
import fuseop

class _FuseBaseModule(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_FuseBaseModule, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
           
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_FuseBaseModule, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _FuseBatchNormRelu(Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight, bias, running_mean, running_var, training, momentum, eps, track_running_stats, num_batches_tracked) -> Tensor:
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

        forward_data = fuseop.bnrelu_forward(
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
            eps
        )
        grad_out = forward_data[0]
        save_mean = forward_data[1]
        save_invstd = forward_data[2]
        ctx.eps = eps
        ctx.bn_training = bn_training
        ctx.save_for_backward(input, weight, save_mean, save_invstd, running_mean, running_var)
        return grad_out
    
    @staticmethod
    def backward(ctx, gradOutput) -> Tensor:
        save_input, weight, save_mean, save_invstd, running_mean, running_var = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fuseop.bnrelu_backward(
            gradOutput, 
            save_input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            ctx.bn_training,
            ctx.eps,
            ctx.needs_input_grad[0:3])
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

class FuseBatchNormRelu1d(_FuseBaseModule):

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )
    def forward(self, input):
        self._check_input_dim(input)
        return _FuseBatchNormRelu.apply(input, self.weight, self.bias, self.running_mean, self.running_var, 
                                        self.training, self.momentum, self.eps,
                                        self.track_running_stats, self.num_batches_tracked)
    

class FuseBatchNormRelu2d(_FuseBaseModule):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return _FuseBatchNormRelu.apply(input, self.weight, self.bias, self.running_mean, self.running_var,
                                        self.training, self.momentum, self.eps,
                                        self.track_running_stats, self.num_batches_tracked)

class FuseBatchNormRelu3d(_FuseBaseModule):

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return _FuseBatchNormRelu.apply(input, self.weight, self.bias, self.running_mean, self.running_var,
                                        self.training, self.momentum, self.eps,
                                        self.track_running_stats, self.num_batches_tracked)