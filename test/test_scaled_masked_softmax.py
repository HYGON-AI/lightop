import math
import torch
import lightop.scaled_masked_softmax as lightopsoftmax
from lightop.scaled_masked_softmax import AttnMaskType

def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores

def forward_torch_softmax(input, mask, scale):
    if scale is not None:
        input = input * scale
    mask_output = attention_mask_func(input, mask) if mask is not None else input
    probs = torch.nn.Softmax(dim=-1)(mask_output)
    return probs

def test_masked_softmax_backward():
    batch = 2
    attn = 16
    scale_t = torch.tensor([0.5])
    for qlen in [128, 256, 1024, 2048, 4096]:
        for klen in [128, 256, 1024, 2048]:
            torch.manual_seed(1)
            fuse_inputs = torch.normal(0, 2, (batch, attn, qlen, klen), dtype=torch.float16, device='cuda:0').requires_grad_()
            masks = torch.randint(0, 2, (batch, 1, qlen, klen), dtype=torch.bool, device='cuda:0')
            fused_softmax = (
                lightopsoftmax.FusedScaleMaskSoftmax(
                    input_in_fp16=True,
                    input_in_bf16=False,
                    mask_func=attention_mask_func,
                    scale=scale_t[0].item(),
                    softmax_in_fp32=True,
                    attn_mask_type=AttnMaskType.padding,
                    scaled_masked_softmax_fusion=True,
                )
                .cuda()
                .half()
            )

            fused_softmax_output = fused_softmax(
                fuse_inputs,
                masks
            )
            torch.manual_seed(1)
            torch_inputs = torch.normal(0, 2, (batch, attn, qlen, klen), dtype=torch.float16, device='cuda:0').requires_grad_()
            softmax_results_torch = forward_torch_softmax(torch_inputs, masks, scale_t[0].item())

            error_forward = (fused_softmax_output-softmax_results_torch).abs().max()
            assert error_forward < 1e-3, (("forward error! scale:0.5 qlen:{} klen:{} ").format(qlen, klen))

            fused_softmax_output.backward(fused_softmax_output)
            softmax_results_torch.backward(softmax_results_torch)
            error = (fuse_inputs.grad-torch_inputs.grad).abs().max()
            assert error < 1e-3, ("backward error! scale:0.5 qlen:{} klen:{} ").format(qlen, klen)
    print("Success")
if __name__ == "__main__":
    test_masked_softmax_backward()
