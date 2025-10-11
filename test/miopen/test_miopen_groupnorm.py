import lightop
import torch
from lightop import miopenGroupNorm

num_groups = 4
batch_size = 1
channel = 4
height = 7
width = 7
total_elements = batch_size * channel * height * width
sequence = torch.arange(1, total_elements + 1,
                        dtype=torch.float,
                        device="cuda")

# 重塑为 NCHW 布局
input = sequence.reshape(batch_size, channel, height, width)
print("input is:", input, input.shape, input.stride())

weight = torch.ones(1, 4, 1, 1).cuda()
print('weight is:', weight, weight.shape, weight.stride())

bias = torch.randn(1, 4, 1, 1).cuda()

dy = sequence.reshape(batch_size, channel, height, width)
print("dy is:", dy, dy.shape, dy.stride())

mean = torch.zeros(4, 1, 1, 1).cuda()
print("mean is:", mean, mean.shape, mean.stride())

rstd = torch.ones(1, 4, 1, 1).cuda()
print("rstd is:", rstd, rstd.shape, rstd.stride())

# 如果要测试 NHWC 布局 则开启以下两行内容, 否则关闭
input_nhwc = input.to(memory_format=torch.channels_last)
dy_nhwc = dy.to(memory_format=torch.channels_last)

output = lightop.miopen_groupnorm_backward(
    input_nhwc, dy_nhwc, weight, mean, rstd, num_groups, epsilon=0.00001, mode=1
)
torch.cuda.synchronize()
print("miopen shape is:", output[0].shape, output[0].stride())

# NCHW 布局下, miopen的gn算子的第一个输出以这个为准
print("miopen bwn nchw output is: =============\n", output[0])

# NHWC 布局下, miopen的gn算子的第一个输出以这个为准
output_nhwc = output[0].permute(0, 2, 3, 1).contiguous(memory_format=torch.channels_last)
print("miopen nhwc shape is:", output_nhwc.shape, output_nhwc.stride())
print("miopen bwn nhwc output is: =============\n", output_nhwc)

# 以下两个输出为1维张量, 数值对上即可
print("miopen shape1 is:", output[1].shape, output[1].stride())
print("miopen bwn output1 is: =============\n", output[1])
print("miopen shape2 is:", output[2].shape, output[2].stride())
print("miopen bwn output2 is: =============\n", output[2])


# NCHW、NHWC(不变)
output_cu = lightop.groupnorm_backward(input, dy, weight, mean, rstd)
out_cu = output_cu[0]
out_1 = output_cu[1]
out_2 = output_cu[2]

print("lightop shape is:", out_cu.shape, out_cu.stride())
# NCHW 布局下, lightop自带的gn算子的第一个输出以这个为准
print("lightop bwn nchw output is: ============\n", out_cu)

# 以下两个输出为1维张量, 数值对上即可
print("lightop nchw shape1 is:", out_1.shape, out_1.stride())
print("lightop bwn output1 is: ============\n", out_1)
print("lightop nchw shape2 is:", out_2.shape, out_2.stride())
print("lightop bwn output2 is: ============\n", out_2)

# NHWC 布局下, lightop自带的gn算子的第一个输出以这个为准
out_cu_nhwc = out_cu.permute(0, 2, 3, 1).contiguous(memory_format=torch.channels_last)
print("lightop nhwc shape is:", out_cu_nhwc.shape, out_cu_nhwc.stride())
print("lightop nhwc bwn output is: ============\n", out_cu_nhwc)

print("=============== GroupNorm Backward End =============== \n")


output = lightop.miopen_groupnorm(
    input, weight, bias, num_groups, epsilon=0.00001, mode=0
)
print("forward result:", output[0])
print("=============== GroupNorm Forward End =============== \n")



add_tensor = torch.randn(batch_size, channel, height, width).cuda()
output = lightop.miopen_add_groupnorm(
    x1=input, x2=add_tensor, weight=weight, bias=bias, num_groups=num_groups, epsilon=0.00001, mode=0
)
torch.cuda.synchronize()
print("forward result:", output[0])
print("=============== Add GroupNorm Forward End =============== \n")