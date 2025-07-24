import lightop
import torch

# 定义张量维度
num_tokens = 1024
hidden_size = 8192
epsilon = 1e-5

# 生成随机输入数据 (bfloat16)
x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16).cuda()
x2 = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16).cuda()

# 生成权重和平滑缩放因子 (可以是bfloat16或float32)
weight = torch.randn(hidden_size, dtype=torch.bfloat16).cuda()  # 测试float32版本
smoothScale = torch.randn(hidden_size, dtype=torch.bfloat16).cuda()  # 测试bfloat16版本

# 调用被测函数
after_res, y, per_token_scale = lightop.add_rms_norm_dynamic_quant(x, x2, weight, smoothScale, epsilon)

torch.cuda.synchronize()
print(y)
