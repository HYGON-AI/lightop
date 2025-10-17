import lightop
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

#query_states shape is : torch.Size([1, 12, 77, 64])
#query_states shape is : torch.Size([1, 20, 77, 64])
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


Q = torch.randn(4, 10, 4096, 64).cuda().half()
K = torch.randn(4, 10, 4096, 64).cuda().half()
V = torch.randn(4, 10, 4096, 64).cuda().half()
for i in range(10):
    out = lightop.miopen_scaled_dot_product_attention(Q, K, V)
torch.cuda.synchronize()
print("===================================================================================================================")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    out = lightop.miopen_scaled_dot_product_attention(Q, K, V)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total"))
torch.cuda.synchronize()
print(out)

for i in range(10):
    output = F.scaled_dot_product_attention(Q,K,V,None,0.0,False)

torch.cuda.synchronize()
print("===================================================================================================================")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    output = F.scaled_dot_product_attention(Q,K,V,None,0.0,False)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total"))
torch.cuda.synchronize()
print(output)

print("test_mha: ", allclose(out, output, rtol=1e-02, atol=1e-02))