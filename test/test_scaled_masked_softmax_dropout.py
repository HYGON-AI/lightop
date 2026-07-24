import math
import torch
import time
from lightop import AttnMaskType
from lightop.attention import FusedScaleMaskSoftmax, FusedScaleMaskSoftmaxDropout


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores

def forward_torch_softmax(input, mask, scale):
    if scale is not None:
        input = input * scale
    mask_output = attention_mask_func(input, mask) if mask is not None else input
    probs = torch.nn.Softmax(dim=-1)(mask_output)
    return probs

def test_time(batch, attn, seq_len, key_len):
    scale_t = torch.tensor([0.5])
    torch.manual_seed(1)
    fuse_input = torch.randn(batch, attn, seq_len,key_len).cuda().half().requires_grad_()
    masks = torch.randint(0, 2, (batch, 1, seq_len, key_len), dtype=torch.bool, device='cuda:0')
    fuse_time_start = time_synchronized()
    for i in range(1000):
        if (i == 10):
            fuse_time_start = time_synchronized()
        torch.manual_seed(1)
        fused_softmax = (
            FusedScaleMaskSoftmaxDropout(
                input_in_fp16=True,
                input_in_bf16=False,
                mask_func=attention_mask_func,
                scale=scale_t[0].item(),
                softmax_in_fp32=True,
                attn_mask_type=AttnMaskType.padding,
                scaled_masked_softmax_fusion=True,
                p = 0.5
            )
            .cuda()
            .half()
        )

        fused_softmax_output = fused_softmax(
            fuse_input,
            masks,
        )
        fused_softmax_output.backward(fused_softmax_output)
    fuse_time_end = time_synchronized()
    print("===fuse:  ",  fuse_time_end - fuse_time_start)

    torch.manual_seed(1)
    torch_attn_weights = torch.randn(batch, attn,seq_len,key_len).cuda().half().requires_grad_()
    time_start = time_synchronized()
    for i in range(1000):
        if (i == 10):
            time_start = time_synchronized()
        src_softmax = (
            FusedScaleMaskSoftmax(
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

        src_softmax_output = src_softmax(
            torch_attn_weights,
            masks,
        )

        dropout = torch.nn.Dropout(p=0.5).cuda()
        src_softmax_output = dropout(src_softmax_output)
        src_softmax_output.backward(src_softmax_output)
    time_end = time_synchronized()
    print("===src:  ", time_end - time_start)

def test_masked_softmax_backward():
    batch = 2
    attn = 16
    scale_t = torch.tensor([0.5])
    for p in [0, 0.2,0.5,0.8,1]:
        for qlen in [128, 256, 1024, 2048, 4096]:
            for klen in [128, 256, 1024, 2048, 4096]:
                torch.manual_seed(1)
                fuse_inputs = torch.normal(0, 2, (batch, attn, qlen, klen), dtype=torch.float16, device='cuda:0').requires_grad_()
                masks = torch.randint(0, 2, (batch, 1, qlen, klen), dtype=torch.bool, device='cuda:0')
                torch.manual_seed(1)
                fused_softmax = (
                    FusedScaleMaskSoftmaxDropout(
                        input_in_fp16=True,
                        input_in_bf16=False,
                        mask_func=attention_mask_func,
                        scale=scale_t[0].item(),
                        softmax_in_fp32=True,
                        attn_mask_type=AttnMaskType.padding,
                        scaled_masked_softmax_fusion=True,
                        p = p
                    )
                    .cuda()
                    .half()
                )

                fused_softmax_output = fused_softmax(
                    fuse_inputs,
                    masks
                )
                if (p != 0):
                    fused_softmax_output += 1
                torch.manual_seed(1)
                src_inputs = torch.normal(0, 2, (batch, attn, qlen, klen), dtype=torch.float16, device='cuda:0').requires_grad_()
                src_softmax = (
                    FusedScaleMaskSoftmax(
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
                
                src_softmax_output = src_softmax(
                    src_inputs,
                    masks
                )
                torch.manual_seed(1)
                dropout = torch.nn.Dropout(p=p).cuda()
                src_softmax_output = dropout(src_softmax_output)
                if (p != 0):
                    src_softmax_output += 1
                #use_half：rtol=1e-02, atol=1e-02,use_float32:rtol=1e-05, atol=1e-05
                print(allclose(fused_softmax_output,src_softmax_output,rtol=2e-02, atol=2e-02))

                fused_softmax_output.backward(fused_softmax_output)
                src_softmax_output.backward(src_softmax_output)
                #use_half：rtol=1e-02, atol=1e-02,use_float32:rtol=1e-05, atol=1e-05
                print(allclose(fuse_inputs.grad,src_inputs.grad,rtol=2e-02, atol=2e-02))
    
    print("Success")
if __name__ == "__main__":
    test_masked_softmax_backward()
    test_time(2,16,128,128)
    # test_time(2,16,256,256)
    # test_time(2,16,1024,1024)
    # test_time(2,16,2048,2048)
    # test_time(2,16,4096,4096)
    # test_time(1,8,2048,2048)
