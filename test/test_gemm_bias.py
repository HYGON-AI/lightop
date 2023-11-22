#https://zhuanlan.zhihu.com/p/372663283
import numpy as np
import torch
import torch.nn.functional as F
import unittest

import time

print("skip test because this op is not performant or convenient")
exit()
from lightop.fuselinearbias import LinearBias

#use_half：rtol=1e-02, atol=1e-02,use_float32:rtol=1e-05, atol=1e-05
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

# m = 107   batch_size*seq_len
# k = 1024  hidden_size
# n = 1024  hidden_size*3/4
shapes = [
    # (2, 3, 4),
    (1440, 2048, 5760),
    (2048, 4320, 5760), 
    (4096, 1536, 1024), 
    (30528, 104, 1024),
]
dims = [3]
warmup_iters = 5
num_iters = 500
forward_only = False
profiling = False

# class Parameter:
#     def __init__(self, w):
#         self.data = np.float32(w)  # 权重
#         self.grad = None  # 梯度
#         self.shape = self.data.shape

# class Fc:
#     def __init__(self, in_features, out_features, bias=True):
#         self.in_features = in_features  # 输入神经元个数
#         self.out_features = out_features  # 输出神经元个数
#         #self.weight = Parameter(np.random.randn(self.in_features, self.out_features) * (2 / self.in_features ** 0.5))
#         self.weight = Parameter(np.random.randn(self.in_features, self.out_features))
#         if bias:
#             #self.bias = Parameter(np.flipud(np.random.randn(self.out_features)))
#             self.bias = Parameter(np.random.randn(self.out_features))
#             # print(self.bias.data)
#         else:
#             self.bias = None
#         self.x_shape = None # 后向过程需要的变量保存起来

#     def forward(self, x):
#         self.x_shape = x.shape  # 记录输入数据的形状
#         self.x = x
#         x = np.dot(x, self.weight.data)
#         if self.bias is not None:
#             x = x + self.bias.data
#         return x
    
#     def backward(self, grad_output):
#         N, _ = grad_output.shape
#         # 计算w的梯度
#         self.weight.grad = np.dot(self.x.T, grad_output)

#         if self.bias is not None:
#             self.bias.grad = np.sum(grad_output, axis=0)
        
#         # 计算传到下一层的梯度
#         x_grad = np.dot(grad_output, self.weight.data.T)
#         return np.reshape(x_grad, self.x_shape)

def set_rand_seed(seed=1):
    #print("Random Seed: ", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False       
    # torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True   # 保证每次返回得的卷积算法是确定的

class FusedLinearTest(unittest.TestCase):
    def setUp(self, seed=1234):
        set_rand_seed(seed)

    def test_performance_half(self):
        pass


for dim in dims:
    for m, n, k in shapes:
        print('m:{m} n:{n} k:{k} input_dim:{dim} forward_only:{forward_only}'.format(m=m, n=n, k=k, dim=dim, forward_only=forward_only))
        set_rand_seed(1234)
        if dim == 2:
            x = np.random.randn(m, k)
            out_grad = np.random.rand(m,n)
            # x = np.ones((m,k))
            # out_grad = np.ones((m,n))
        elif dim == 3:
            x = np.random.randn(m,1,k)
            out_grad = np.random.rand(m,1,n)
        # x = np.ones((m,k)) + 1
        # fc_layer = Fc(k, n, bias=True)
        # y = fc_layer.forward(x)
        # out_grad = np.ones((m,n))
        # print('x:', x)
        # np.savetxt('x.txt', x)
        # print('out_grad:', out_grad)
        # np.savetxt('out_grad.txt', out_grad)
        # grad = fc_layer.backward(out_grad)
        # print(y)

        x1 = torch.from_numpy(x).float().cuda(0).half()
        x1.requires_grad = True
        x1.retain_grad()
        out_grad_cuda1 = torch.from_numpy(out_grad).cuda(0).half()

        x2 = torch.from_numpy(x).float().cuda(0).half()
        x2.requires_grad = True
        x2.retain_grad()
        out_grad_cuda2 = torch.from_numpy(out_grad).cuda(0).half()

        set_rand_seed(1234)
        fc1 = torch.nn.Linear(k, n, bias=True, device="cuda:0", dtype=torch.float16)

        # warmup
        for i in range(warmup_iters):
            out = fc1(x1)
            if not forward_only:
                out.backward(out_grad_cuda1)
        torch.cuda.synchronize()

        # if not (m == 1024 and n == 101 and k == 1024):
        if not profiling:
            start_time = time.time()
            for i in range(num_iters):
                out1 = fc1(x1)
                if not forward_only:
                    out1.backward(out_grad_cuda1)
                torch.cuda.synchronize()
            end_time = time.time()
            fc1_espl = end_time - start_time
            print("origin fc bias time is ", fc1_espl, 'averange time is ', fc1_espl/num_iters)
        else:
            with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, use_kineto=True) as prof:
                for i in range(num_iters):
                    out1 = fc1(x1)
                    if not forward_only:
                        out1.backward(out_grad_cuda1)
                    time.sleep(0.1)
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="self_cuda_time_total"))
            prof.export_chrome_trace("trace_origin.json")

        set_rand_seed(1234)
        fc2 = LinearBias(k, n, bias=True, device="cuda:0", dtype=torch.float16)

        # warmup
        for i in range(warmup_iters):
            out = fc2(x2)
            if not forward_only:
                out.backward(out_grad_cuda2)
        torch.cuda.synchronize()

        # if not (m == 1024 and n == 101 and k == 1024):
        if not profiling:
            start_time = time.time()
            for i in range(num_iters):
                out2 = fc2(x2)
                if not forward_only:
                    out2.backward(out_grad_cuda2)
                torch.cuda.synchronize()
            end_time = time.time()
            fc2_espl = end_time - start_time
            print("fused fc bias time is ", fc2_espl, 'averange time is ', fc2_espl/num_iters, ', ', (fc1_espl-fc2_espl)*100/fc1_espl)
        else:
            with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, use_kineto=True) as prof:
                for i in range(num_iters):
                    out2 = fc2(x2)
                    if not forward_only:
                        out2.backward(out_grad_cuda2)
                    time.sleep(0.1)
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="self_cuda_time_total"))
            prof.export_chrome_trace("trace_fused.json")
        # print(np.linalg.norm(out1.detach().cpu().numpy()-out2.detach().cpu().numpy()))
        #print(np.linalg.norm(fc1.weight.detach().cpu().numpy()-fc2.weight.detach().cpu().numpy()))
        #print(np.linalg.norm(fc1.bias.detach().cpu().numpy()-fc2.bias.detach().cpu().numpy()))

        #print(np.linalg.norm(x1.grad.detach().cpu().numpy()-x2.grad.detach().cpu().numpy()))
        #print(np.linalg.norm(fc1.weight.grad.detach().cpu().numpy()-fc2.weight.grad.detach().cpu().numpy()))
        #print(np.linalg.norm(fc1.bias.grad.detach().cpu().numpy()-fc2.bias.grad.detach().cpu().numpy()))

        #print(np.linalg.norm(y.data-out.detach().cpu().numpy()))
        #print(np.linalg.norm(fc_layer.weight.data.transpose()-fc1.weight.detach().cpu().numpy()))
        #print(np.linalg.norm(fc_layer.bias.data-fc1.bias.detach().cpu().numpy()))
        #
        #print(np.linalg.norm(grad-x.grad.detach().cpu().numpy()))
        #print(np.linalg.norm(fc_layer.weight.grad.transpose()-fc1.weight.grad.detach().cpu().numpy()))
        ##print(np.linalg.norm(fc_layer.weight.grad-fc1.weight.grad.detach().cpu().numpy()))
        #print(np.linalg.norm(fc_layer.bias.grad.data-fc1.bias.grad.detach().cpu().numpy()))
        if forward_only:
            print("test_linearbias:",allclose(out1,out2,rtol=1e-05, atol=1e-05), allclose(fc1.weight,fc2.weight,rtol=1e-05, atol=1e-05), allclose(fc1.bias,fc2.bias,rtol=1e-05, atol=1e-05))
        else:
            print("test_linearbias:",allclose(out1,out2,rtol=1e-05, atol=1e-05), allclose(fc1.weight,fc2.weight,rtol=1e-05, atol=1e-05), allclose(fc1.bias,fc2.bias,rtol=1e-05, atol=1e-05), 
                                     allclose(x1.grad,x2.grad,rtol=1e-05, atol=1e-05), allclose(fc1.weight.grad,fc2.weight.grad,rtol=1e-05, atol=1e-05), allclose(fc1.bias.grad,fc2.bias.grad,rtol=1e-05, atol=1e-05))
                                  
        # print(fc1.bias)
        # print(fc2.bias)
        # print(out1)
        # print(out2)
        # print(fc1.weight)
        # print(fc2.weight)
        # print(x1.grad)
        # print(x2.grad)
        # print(fc1.weight.grad)
        # print(fc2.weight.grad)
        # print(out_grad_cuda1)
        # print(out_grad_cuda2)
        # print(fc1.bias.grad)
        # print(fc2.bias.grad)
        # print(torch.allclose(out1, out2, atol=1e-2))

