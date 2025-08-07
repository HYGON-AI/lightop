import lightop
import torch

# 定义张量维度
epsilon = 1e-5

# 生成随机输入数据 (bfloat16)
x = torch.randn(112, 8, 128, 1, dtype=torch.bfloat16).cuda()
weight = torch.randn(1024, dtype=torch.float).cuda()

# 调用被测函数
y = lightop.miopen_head_rms_norm(x, weight, epsilon, 0, 8, 8)
torch.cuda.synchronize()
print(y)
