#https://zhuanlan.zhihu.com/p/372663283
import numpy as np
import torch
import torch.nn.functional as F

import time

from lightop.fuselinearbias import LinearBias

# m = 107
# k = 1024
# n = 1024
shapes = [
    (2, 16, 1024), 
    (107, 1024, 1024), 
    (1024, 2, 1024), 
    (1024, 101, 1024), 
    (4096, 1536, 1024), 
    (30528, 104, 1024),
]

class Parameter:
    def __init__(self, w):
        self.data = np.float32(w)  # 权重
        self.grad = None  # 梯度
        self.shape = self.data.shape

class Fc:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features  # 输入神经元个数
        self.out_features = out_features  # 输出神经元个数
        #self.weight = Parameter(np.random.randn(self.in_features, self.out_features) * (2 / self.in_features ** 0.5))
        self.weight = Parameter(np.random.randn(self.in_features, self.out_features))
        if bias:
            #self.bias = Parameter(np.flipud(np.random.randn(self.out_features)))
            self.bias = Parameter(np.random.randn(self.out_features))
            # print(self.bias.data)
        else:
            self.bias = None
        self.x_shape = None # 后向过程需要的变量保存起来

    def forward(self, x):
        self.x_shape = x.shape  # 记录输入数据的形状
        self.x = x
        x = np.dot(x, self.weight.data)
        if self.bias is not None:
            x = x + self.bias.data
        return x
    
    def backward(self, grad_output):
        N, _ = grad_output.shape
        # 计算w的梯度
        self.weight.grad = np.dot(self.x.T, grad_output)

        if self.bias is not None:
            self.bias.grad = np.sum(grad_output, axis=0)
        
        # 计算传到下一层的梯度
        x_grad = np.dot(grad_output, self.weight.data.T)
        return np.reshape(x_grad, self.x_shape)

def set_rand_seed(seed=1):
    #print("Random Seed: ", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False       
    # torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True   # 保证每次返回得的卷积算法是确定的

set_rand_seed(1234)
forward_only = True
for dim in (2, 3):
    for is_bias in (False, True):
        for m, n, k in shapes:
            print('m:{m} n:{n} k:{k} input_dim:{dim} is_bias:{is_bias}'.format(m=m, n=n, k=k, dim=dim, is_bias=is_bias))
            if dim == 2:
                x = np.random.randn(m, k)
                out_grad = np.random.rand(m,n)
            elif dim == 3:
                x = np.random.randn(m, 1, k)
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
            fc1 = torch.nn.Linear(k, n, bias=is_bias, device="cuda:0", dtype=torch.float16)

            # warmup
            for i in range(5):
                out = fc1(x1)
                # out.backward(out_grad_cuda1)

            torch.cuda.synchronize()
            start_time = time.time()
            #with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True) as prof:
            for i in range(10000):
                out1 = fc1(x1)
                # out1.backward(out_grad_cuda1)
            torch.cuda.synchronize()
            #print(prof.key_averages().table(sort_by="self_cuda_time_total"))
            #prof.export_chrome_trace("trace.json")
            end_time = time.time()
            print("origin fc bias time is ", end_time - start_time)


            set_rand_seed(1234)
            fc2 = LinearBias(k, n, bias=is_bias, device="cuda:0", dtype=torch.float16)
            # warmup
            for i in range(5):
                out = fc2(x2)
                # out.backward(out_grad_cuda2)

            torch.cuda.synchronize()
            start_time = time.time()
            #with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True) as prof:
            for i in range(10000):
                out2 = fc2(x2)
                # out2.backward(out_grad_cuda2)
            torch.cuda.synchronize()
            #print(prof.key_averages().table(sort_by="self_cuda_time_total"))
            #prof.export_chrome_trace("trace.json")
            end_time = time.time()
            print("fused fc bias time is ", end_time - start_time)

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
            if is_bias:
                if forward_only:
                    print("test_linearbias:",out1.equal(out2) and fc1.weight.equal(fc2.weight) and fc1.bias.equal(fc2.bias))
                else:
                    print("test_linearbias:", out1.equal(out2) and fc1.weight.equal(fc2.weight) and fc1.bias.equal(fc2.bias) and x1.grad.equal(x2.grad) and fc1.weight.grad.equal(fc2.weight.grad) and fc1.bias.grad.equal(fc2.bias.grad))
            else:
                if forward_only:
                    print("test_linearbias:", out1.equal(out2) and fc1.weight.equal(fc2.weight))
                else:
                    print("test_linearbias:", out1.equal(out2), fc1.weight.equal(fc2.weight), x1.grad.equal(x2.grad), fc1.weight.grad.equal(fc2.weight.grad))
            # print(torch.allclose(out1, out2, atol=1e-2))
            # print(fc1.bias.grad, fc1.bias.grad.shape)
            # print(fc2.bias.grad)
