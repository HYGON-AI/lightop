import torch
import time
import torch.nn.functional as F
from torch.autograd import Variable
from lightop import conv_add

# 检验计算结果存在的误差是否可接受
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):

    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

# 初始化数据
def get_data(batch_size, in_channels_per_group, H, W, use_half = False):

    if use_half:
        input = torch.rand((batch_size, in_channels_per_group, H, W)).cuda().half()
        add = torch.rand((batch_size, in_channels_per_group, H, W)).cuda().half()
        weight = torch.rand((batch_size, in_channels_per_group, H, W)).cuda().half()
        
    else:
        input = torch.rand((in_channels_per_group, batch_size, H, W)).cuda()
        add  = torch.rand((batch_size, in_channels_per_group, H, W)).cuda()
        weight = torch.rand((batch_size, in_channels_per_group, H, W)).cuda().half()
   
    x1 = input.clone().detach()
    x2 = add.clone().detach()
    x3 = weight.clone().detach()
    
    return x1, x2, x3

# 获取模型
def get_model(batch_size, channels, out_channels, kernel_size, stride, padding, dilation, layout, use_half=False):

    if use_half:
        fuse_conv_add=conv_add(in_channels = channels, out_channels = out_channels,
                                  kernel_size = kernel_size, stride = stride, padding = padding,
                                  dilation = dilation, layout = layout).cuda().half()
        
    else:
        fuse_conv_add=conv_add(in_channels = channels, out_channels = out_channels,
                                  kernel_size = kernel_size, stride = stride, padding = padding,
                                  dilation = dilation, layout = layout).cuda()
    
    return fuse_conv_add

def test_speed(batch_size, channels, H, W, out_channels_per_group,
               groups, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation,
               layout, repeat_num=10,use_half=False):


    x1,x2, weight = get_data(batch_size, channels, H, W, use_half=use_half)
    
    out_channels = out_channels_per_group * groups
    kernel_size = (kernel_h, kernel_w)
    stride = (stride_h, stride_w)
    padding = (pad_h, pad_w)
    dilation = (dilation, dilation)
    
    f1 = get_model(batch_size, channels, out_channels, kernel_size, stride, 
                   padding, dilation, layout, use_half=use_half)

    torch.manual_seed(0)
    start_time = time.time()
    for i in range(repeat_num):
        y1 = f1(x1, x2, weight)
        torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time
    print('torch_bias_dropout_add', 'averange time is', fc1_espl *1000/ repeat_num, 'ms')

    # torch.manual_seed(0)
    # start_time = time.time()
    # for i in range(repeat_num):
    #     bias2.grad = None
    #     x2.grad = None
    #     residual2.grad = None
    #     y2 = f2(x2, bias2,residual2)
    #     y2.backward(grad)
    #     torch.cuda.synchronize()
    # end_time = time.time()    
    # fc2_espl = end_time - start_time
    # print('fuse_bias_dropout_add', 'averange time is', fc2_espl *1000/ repeat_num, 'ms')

    # #误差计算
    # if use_half:
    #     print(allclose(y1,y2,rtol=2e-02, atol=2e-02))
    #     # print(allclose(x1.grad,x2.grad,rtol=2e-02, atol=2e-02))
    #     # print(allclose(bias1.grad,bias2.grad,rtol=2e-02, atol=2e-02))
    # else:
    #     print(allclose(y1,y2,rtol=1e-05, atol=1e-05))
    #     # print(allclose(x1.grad,x2.grad,rtol=1e-05, atol=1e-05))
    #     # print(allclose(bias1.grad,bias2.grad,rtol=1e-05, atol=1e-05))


if __name__ == "__main__":

    #设置基础参数
    # batchsize, seqs, cols = 1, 2048, 3072
    batch_size = 2
    channels = 2
    H = 8
    W = 8
    out_channels_per_group = 2
    groups = 3
    kernel_h = 3
    kernel_w = 3
    stride_h = 2
    stride_w = 2
    pad_h = 1
    pad_w = 1
    dilation = 1
    layout = "NCHW"
    #耗时测试
    test_speed(batch_size, channels, H, W, out_channels_per_group,
               groups, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation,
               layout, repeat_num=10, use_half=False)
    #性能分析
    #test_profile(batchsize, seqs, cols,dropout_p,repeat_num=100,use_half=False)