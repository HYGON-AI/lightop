import torch
import time
from lightop.fuse_bias_swiglu import FuseBiasSwiGLU
import torch.nn.functional as F

if __name__ == '__main__':

    RUNS = 1000

    batch_size = 2
    seq_len = 2048
    num_channels = 1024
    x = torch.randn(batch_size, seq_len, num_channels).cuda().half()
    bias = torch.randn(num_channels).cuda().half()
    x.requires_grad_(True)
    bias.requires_grad_(True)
    # glu should halve the last dimension
    output_shape = (batch_size, seq_len, num_channels // 2)
    grad = torch.randn(output_shape).cuda().half()
    fuse_bias_swiglu = FuseBiasSwiGLU().cuda().half()

    for _ in range(RUNS):
        fused = fuse_bias_swiglu(x, bias)
        x_in = x + bias
        input, mul = x_in.chunk(2, dim=-1)
        expected = mul * F.silu(input)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(RUNS):
        x.grad = None
        bias.grad = None
        fused = fuse_bias_swiglu(x, bias)
        fused.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()
    x_grad_fused, bias_grad_fused = x.grad, bias.grad
    print(f"custom_kernel time {(end_time - start_time)*1000/RUNS} ms")
    
    start_time = time.time()
    for _ in range(RUNS):
        x.grad = None
        bias.grad = None
        x_in = x + bias
        input, mul = x_in.chunk(2, dim=-1)
        expected = mul * F.silu(input)
        expected.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()
    x_grad_expected, bias_grad_expected = x.grad, bias.grad
    
    print(f"jit time {(end_time - start_time)*1000/RUNS} ms")

    def rerr(x1, x2):
        return ((x1.float() - x2.float()) / (x1.float() + x2.float() + 1e-6)).abs().max()

    print(((expected - fused).abs().max()))
    print(((x_grad_expected - x_grad_fused).abs().max()))
    print(((bias_grad_expected - bias_grad_fused).abs().max()))

    print(
        f"element-wise relative error max: output: {rerr(expected, fused)}")
    print(
        f"element-wise relative error max: x.grad: {rerr(x_grad_expected, x_grad_fused)}, bias.grad: {rerr(bias_grad_expected, bias_grad_fused)}")
 
