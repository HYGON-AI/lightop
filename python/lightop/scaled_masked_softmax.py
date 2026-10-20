# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import enum
import torch.nn as nn
#from lightop import op
from . import op

class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2

# class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
#     """
#     Fused operation which performs following three operations in sequence
#     1. Scale the tensor.
#     2. Apply upper triangular mask (typically used in gpt models).
#     3. Perform softmax.
#     """

#     @staticmethod
#     def forward(ctx, inputs, scale):
#         scale_t = torch.tensor([scale])
#         softmax_results = op.forward(
#             inputs, scale_t[0]
#         )

#         ctx.save_for_backward(softmax_results, scale_t)
#         return softmax_results
#     @staticmethod
#     def backward(ctx, output_grads):
#         softmax_results, scale_t = ctx.saved_tensors
#         input_grads = op.backward(
#             output_grads, softmax_results, scale_t[0]
#         )

#         return input_grads, None

class FusedScaleMaskSoftmax(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
    ):
        super(FusedScaleMaskSoftmax, self).__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        assert not (
            self.input_in_fp16 and self.input_in_bf16
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        assert (
            self.scale is None or softmax_in_fp32
        ), "softmax should be in fp32 when scaled"

    def forward(self, input, mask):
        # [b, np, sq, sk]
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
        
        if self.is_kernel_available(mask, *input.size()):
            # print("use fuse kernel")
            return self.forward_fused_softmax(input, mask)
        else:
            # print("no use fuse kernel, use torch")
            return self.forward_torch_softmax(input, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np

        if (
            self.scaled_masked_softmax_fusion  # user want to fuse
            and self.input_in_float16  # input must be fp16
            and 16 < sk <= 4096  # sk must be 16 ~ 2048
            and sq % 4 == 0  # sq must be divisor of 4
            and sk % 4 == 0  # sk must be divisor of 4 
            and attn_batches % 4 == 0  # np * b must be divisor of 4
        ):
            if 0 <= sk <= 4096:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)

                if self.attn_mask_type == AttnMaskType.causal:
                    if attn_batches % batch_per_block == 0:
                        return True
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def forward_fused_softmax(self, input, mask):
        b, np, sq, sk = input.size()
        scale = self.scale if self.scale is not None else 1.0

        if self.attn_mask_type == AttnMaskType.causal:
            assert sq == sk, "causal mask is only for self attention"
            # input is 3D tensor (attn_batches, sq, sk)
            input = input.view(-1, sq, sk)
            if input.dim() != 3:
                raise ValueError("expected 3D input (got {}D input)".format(input.dim()))
            probs = op.scaled_upper_triang_masked_softmax_forward_autograd(input, scale)
            return probs.view(b, np, sq, sk)
        else:
            # raise ValueError("causal mask is only for self attention")
            # input is 4D tensor (b, np, sq, sk)
            if input.dim() != 4:
                raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
            if mask is not None:
                if mask.dim() != 4:
                    raise ValueError("expected 4D mask (got {}D input)".format(mask.dim()))
                return op.scaled_masked_softmax_forward_autograd(input, mask, scale)
            else:
                raise ValueError("nead mask")
                # return ScaledSoftmax.apply(input, scale)

    def forward_torch_softmax(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale
        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        return op.get_batch_per_block(sq, sk, b, np)


class FusedScaleMaskSoftmaxDropout(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
        p: float = 0.5
    ):
        super(FusedScaleMaskSoftmaxDropout, self).__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        assert not (
            self.input_in_fp16 and self.input_in_bf16
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        self.p = p
        assert (0 <= self.p <= 1), "dropout p is 0---1"
        assert (
            self.scale is None or softmax_in_fp32
        ), "softmax should be in fp32 when scaled"

    def forward(self, input, mask):
        # [b, np, sq, sk]
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
        
        if self.is_kernel_available(mask, *input.size()):
            # print("use fuse kernel")
            return self.forward_fused_softmax_dropout(input, mask)
        else:
            # print("no use fuse kernel, use torch")
            return self.forward_torch_softmax_dropout(input, mask)

    def extra_repr(self):
        return f'p={round(self.p,3):0.3f}'

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np
        if (
            self.scaled_masked_softmax_fusion  # user want to fuse
            and self.input_in_float16  # input must be fp16
            and 16 < sk <= 4096  # sk must be 16 ~ 2048
            and sq % 4 == 0  # sq must be divisor of 4
            and sk % 4 == 0  # sk must be divisor of 4 
            and attn_batches % 4 == 0  # np * b must be divisor of 4
        ):
            if 0 <= sk <= 4096:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)
                if self.attn_mask_type == AttnMaskType.causal:
                    if attn_batches % batch_per_block == 0:
                        return True
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def forward_fused_softmax_dropout(self, input, mask):
        b, np, sq, sk = input.size()
        scale = self.scale if self.scale is not None else 1.0
        p = self.p
        if self.attn_mask_type == AttnMaskType.causal:
            assert sq == sk, "causal mask is only for self attention"
            # input is 3D tensor (attn_batches, sq, sk)
            input = input.view(-1, sq, sk)
            if input.dim() != 3:
                raise ValueError("expected 3D input (got {}D input)".format(input.dim()))
            probs = op.scaled_upper_triang_masked_softmax_dropout_forward_autograd(input, scale, p)
            return probs.view(b, np, sq, sk)
        else:
            # raise ValueError("causal mask is only for self attention")
            # input is 4D tensor (b, np, sq, sk)
            if input.dim() != 4:
                raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
            if mask is not None:
                if mask.dim() != 4:
                    raise ValueError("expected 4D mask (got {}D input)".format(mask.dim()))
                return op.scaled_masked_softmax_dropout_forward_autograd(input, mask, scale, p)
            else:
                raise ValueError("nead mask")
                # return ScaledSoftmax.apply(input, scale)

    def forward_torch_softmax_dropout(self, input, mask, p):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale
        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)
        
        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()
                
        dropout = torch.nn.Dropout(p=p).cuda()
        probs = dropout(probs)
        return probs

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        return op.get_batch_per_block(sq, sk, b, np)
