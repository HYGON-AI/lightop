import numbers

import torch
from torch.nn import init
from torch.nn.parameter import Parameter

from lightop import op


class FusedLnRmsNormsAddDropout(torch.nn.Module):

    def __init__(self, norm_type: str = 'ln', normalized_shape: int = 1, normalized_eps: float = 1e-5, normalized_affine: bool = True,
                 bias_type='bias', drop_type: str = 'dropout', drop_prob: float = 0.5):
        """
            .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
            .. _`RMS Normalization`: https://arxiv.org/pdf/1910.07467

            super(FusedNormAddDrop, self).__init__()
            Op( drop(x) + bias )

            :param norm_type: options:  none / ln / rms
            :param bias_type: options:  none / bias
            :param drop_type: options:  none / dropout
            :param drop_prob: rate
        """
        super(FusedLnRmsNormsAddDropout, self).__init__()

        # #### norm init ####
        self.norm_type_enum = self.__get_type_enum(norm_type, ['none', 'ln', 'rms'])

        self.normalized_shape = normalized_shape
        self.normalized_eps = normalized_eps
        self.normalized_affine = normalized_affine

        if norm_type == 'ln':
            if self.normalized_affine:
                self.weight = Parameter(torch.Tensor(normalized_shape))
                self.bias = Parameter(torch.Tensor(normalized_shape))
                init.ones_(self.weight)
                init.zeros_(self.bias)
            else:
                self.weight = torch.Tensor()
                self.bias = torch.Tensor()
        elif norm_type == 'rms':
            if self.normalized_affine:
                self.weight = Parameter(torch.Tensor(normalized_shape))
                init.ones_(self.weight)
            else:
                self.weight = torch.Tensor()
            self.bias = torch.Tensor()
        else:
            self.weight = torch.Tensor()
            self.bias = torch.Tensor()

        # #### bias drop init ####
        self.bias_type_enum = self.__get_type_enum(bias_type, ['none', 'bias'])
        self.drop_type_enum = self.__get_type_enum(drop_type, ['none', 'dropout'])
        self.drop_prob = drop_prob
        self.drop_mask = None

    @staticmethod
    def __get_type_enum(type_name, type_available: list):
        if type_name not in type_available:
            raise ValueError('cannot find {} from {}'.format(type_name, type_available))
        return type_available.index(type_name)

    def forward(self, x, residual=torch.Tensor()):
        out, self.drop_mask = op.lnrmsnorms_add_drop_forward_autograd(
            x, self.weight, self.bias,
            self.normalized_shape, self.normalized_eps, self.normalized_affine, residual, self.drop_prob,
            self.norm_type_enum, self.bias_type_enum, self.drop_type_enum, self.training)
        return out

    def extra_repr(self):
        return "{normalized_shape}, normalized_eps={normalized_eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)
