import lightop
import torch

# 定义张量维度
lr = 0.001
beta1 = 0.9
beta2 = 0.999
weight_decay = 0
eps = 1e-8
amsgrad = False
maximize = False

# 生成随机输入数据 (float)
params = torch.randn(64, 32, 128, 3, dtype=torch.float).cuda()
grads = torch.randn(64, 32, 128, 3, dtype=torch.float).cuda()
exp_avgs = torch.randn(64, 32, 128, 3, dtype=torch.float).cuda()
exp_avg_sqs = torch.randn(64, 32, 128, 3, dtype=torch.float).cuda()
max_exp_avg_sqs = torch.randn(64, 32, 128, 3, dtype=torch.float).cuda()


state_steps = torch.tensor([10], dtype=torch.int32).cuda()


# 调用被测函数
lightop.miopen_adamw(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    amsgrad = amsgrad,
    beta1 = beta1,
    beta2 = beta2,
    lr = lr,
    weight_decay = weight_decay,
    eps = eps,
    maximize = maximize
)

torch.cuda.synchronize()
print("================= After Compute ===================\n")
print(params)

