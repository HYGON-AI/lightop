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

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

print("===================================================================================================================")

# 测试miopen接口，调用miopen api warmup
for i in range(10):
    y2 = lightop.miopen_transform_tensor(input_, 0)

# 统计miopen API耗时
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start_event.record()
for i in range(10):
    y2 = lightop.miopen_transform_tensor(input_, 0)
end_event.record()
torch.cuda.synchronize()
miopen_api_time = start_event.elapsed_time(end_event) / 10
print('Self miopen api time: ', round(miopen_api_time, 3))

# 测试miopen prof耗时
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    out = lightop.miopen_transform_tensor(input_, 0)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total"))

print("out Tensor shape :", out.shape)
