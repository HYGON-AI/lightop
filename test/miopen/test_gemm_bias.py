import lightop
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

a = torch.randn(1, 8192, 640).cuda().half()
b = torch.randn(640, 640).cuda().half()
bias = torch.randn(1, 640).cuda().half()


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

print("===================================================================================================================")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        out = lightop.miopen_gemm_bias(a, b, bias)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total"))
torch.cuda.synchronize()
print(out)
print("out Tensor shape :", out.shape)



print("===================================================================================================================")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    output = torch.matmul(a, b) + bias
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total"))
torch.cuda.synchronize()
print(output)
print("output Tensor shape :", output.shape)

print("test_gemm_add: ", allclose(out, output, rtol=1e-02, atol=1e-02))