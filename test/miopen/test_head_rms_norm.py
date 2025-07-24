import lightop
import torch

# 定义张量维度
epsilon = 1e-5

# 生成随机输入数据 (bfloat16)
x = torch.randn(1, 32, 14, 14, dtype=torch.bfloat16).cuda()
weight = torch.randn(1, 32, 14, 14, dtype=torch.float).cuda()

# 调用被测函数
y = lightop.head_rms_norm(x, weight, epsilon, 0, 32)
torch.cuda.synchronize()
print(y)
