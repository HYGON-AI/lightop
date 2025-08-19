import lightop
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from lightop import miopen_transform_tensor

# input_ = torch.randn(16, 64, 8, 8).cuda().half()

# 计算张量总元素个数 (2*32*2*1)
total_elements = 2 * 32 * 2 * 1

# 创建连续数字的一维张量 [1, 2, 3, ..., 128]
sequence = torch.arange(1, total_elements + 1, 
                        dtype=torch.half, 
                        device="cuda")

# 重塑为目标形状 (2, 32, 2, 1)
input_ = sequence.reshape(2, 32, 2, 1)
input = input_.contiguous(memory_format=torch.channels_last)  # NHWC 格式

# NCHW ---> NCHWc32:10
# out = lightop.miopen_transform_tensor(input_, 0)

# NHWC ---> NCHWc32:10
# out = lightop.miopen_transform_tensor(input, 1)

# NCHWc32 ---> NCHW:0
# out = lightop.miopen_transform_tensor(input_, 2)

# # NCHWc32 ---> NHWC:1
# out = lightop.miopen_transform_tensor(input_, 3)


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

print("===================================================================================================================")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    out = lightop.miopen_transform_tensor(input_, 0)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total"))
torch.cuda.synchronize()
print(out)
print("out Tensor shape :", out.shape)

# print("===================================================================================================================")
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     output = torch.matmul(a, b) + bias
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total"))
# torch.cuda.synchronize()
# print(output)

# print("out is :\n", out)
print("Transform---------------\n")
print("out Tensor shape :", out.shape)

# print("test_gemm_add: ", allclose(out, output, rtol=1e-02, atol=1e-02))