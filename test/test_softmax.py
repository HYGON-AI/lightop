#  -*-  coding: utf-8   -*-
import torch
import time
import copy
from lightop.attention import FuseLogSoftmax, FuseSoftmax
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

device = torch.device('cuda')
torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)


class SoftmaxTester:
    def __init__(self, softmax_type='default', dim=-1, test_half=False, half_to_float=False):
        torch.backends.cudnn.enabled = False
        self.half_to_float = half_to_float
        self.test_half = test_half
        self.softmax_type = softmax_type
        self.input_dtype = torch.float16 if self.test_half else torch.float32

        if softmax_type == 'default' or softmax_type == 'softmax':
            self.torch_softmax = nn.Softmax(dim=dim).cuda()
            self.fuse_softmax = FuseSoftmax(dim=dim, half_to_float=half_to_float).cuda()
        elif softmax_type == 'log_softmax':
            self.torch_softmax = nn.LogSoftmax(dim=dim).cuda()
            self.fuse_softmax = FuseLogSoftmax(dim=dim, half_to_float=half_to_float).cuda()
        else:
            raise NotImplementedError('not support type {}'.format(softmax_type))

    def allclose(self, a, b, equal_nan=False):
        if self.softmax_type in ['default', 'softmax']:
            if self.test_half:
                rtol, atol = 2e-02, 2e-02
            else:
                rtol, atol = 1e-05, 1e-05
        else:
            if self.test_half:
                rtol, atol = 2e-02, 2e-02
            else:
                rtol, atol = 2e-04, 2e-04

        return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def check_accuracy(self, x_size=(4, 60, 60)):
        # 设置输入数据
        x0 = torch.rand(x_size, dtype=self.input_dtype).cuda()
        x = Variable(x0, requires_grad=True)
        x0_fuse = copy.deepcopy(x0).cuda()
        x_fuse = Variable(x0_fuse, requires_grad=True)
        grad = torch.rand(x0.shape, dtype=torch.float32).cuda() * 1e2

        self.fuse_softmax.train()
        self.torch_softmax.train()

        # 检查准确性
        fuse_out = self.fuse_softmax(x_fuse)
        fuse_out.backward(grad)
        torch.cuda.synchronize()

        out = self.torch_softmax(x)
        out.backward(grad)
        torch.cuda.synchronize()
        out = out.to(fuse_out.dtype)

        # print('out: ', out[:2, :2, :5])
        # print('fuse_out: ', fuse_out[:2, :2, :5])
        if self.allclose(out, fuse_out):
            print("forward Passed")
        else:
            print("forward Failed")

        # print('x.grad: ', x.grad[:2, :2, :5])
        # print('x_fuse.grad: ', x_fuse.grad[:2, :2, :5])
        if self.allclose(x.grad, x_fuse.grad):
            print("backward Passed")
        else:
            print("backward Failed")
            diff = torch.abs(x.grad - x_fuse.grad)
            print('diff', diff.shape)
            # print('x.grad: ', x.grad[:3, :3, :5])
            # print('x_fuse.grad: ', x_fuse.grad[:3, :3, :5])
            print(torch.argmax(diff))
            print('diff max={}, mean={}'.format(diff.max(), diff.mean()))

    def test_single_speed(self, softmax_tool, x_size=(4, 60, 60),
                          forward_only=False, repeat_num=1000):
        # 设置输入数据
        x0 = torch.rand(x_size, dtype=self.input_dtype).cuda()
        x = Variable(x0, requires_grad=True)
        grad = torch.rand(x0.shape, dtype=torch.float32).cuda()

        if forward_only:
            softmax_tool.eval()
        else:
            softmax_tool.train()

        # warm up
        for i in range(10):
            out = softmax_tool(x)
            if not forward_only:
                out.backward(grad)
        torch.cuda.synchronize()

        start_time = time.time()
        for i in range(repeat_num):
            out = softmax_tool(x)
            if not forward_only:
                x.grad = None
                out.backward(grad)
        torch.cuda.synchronize()

        end_time = time.time()
        fc1_espl = end_time - start_time

        return fc1_espl * 1000.0 / repeat_num

    def test_single_speed_v2(self, softmax_tool, x_size=(4, 60, 60), repeat_num=1000):
        # 设置输入数据
        x0 = torch.rand(x_size, dtype=self.input_dtype).cuda()
        x = Variable(x0, requires_grad=True)
        grad = torch.rand(x0.shape, dtype=torch.float32).cuda()

        softmax_tool.train()

        # warm up
        for i in range(10):
            out = softmax_tool(x)
            out.backward(grad)
            torch.cuda.synchronize()

        fwd_time = 0.0
        bwd_time = 0.0
        start_time = time.time()
        for i in range(repeat_num):
            out = softmax_tool(x)
            torch.cuda.synchronize()
            fwd_time += time.time() - start_time

            x.grad = None
            mid_time = time.time()
            out.backward(grad)
            torch.cuda.synchronize()
            bwd_time += time.time() - mid_time
            start_time = time.time()

        return fwd_time * 1000.0 / repeat_num, bwd_time * 1000.0 / repeat_num

    def test_speed(self, x_size=(4, 60, 60), forward_only=True, split_fwd_bwd=True, repeat_num=500):
        """测试速度

        Args:
            x_size (tuple, optional): 输入的tensor大小. Defaults to (4, 60, 60).
            forward_only (bool, optional): 是否运行推理模式，只测试前向. Defaults to True.
            split_fwd_bwd (bool, optional): 测试前向反向时是否分别记录前向和反向的耗时. Defaults to 1000.
            repeat_num (int, optional): 测试重复的次数. Defaults to 1000.

        Returns:
            _type_: 耗时统计
        """
        if forward_only:
            torch_fwd_avg_time = self.test_single_speed(self.torch_softmax, x_size, True, repeat_num)
            fuse_fwd_avg_time = self.test_single_speed(self.fuse_softmax, x_size, True, repeat_num)

            fwd_acc_rate = (torch_fwd_avg_time - fuse_fwd_avg_time) / torch_fwd_avg_time * 100.0
            print('torch fwd averange time={:.4f} ms\nfuse fwd averange time={:.4f} ms, accelerate={:.2f}%'.format(
                torch_fwd_avg_time, fuse_fwd_avg_time, fwd_acc_rate
            ))
            print('=' * 30, '\n')
            return torch_fwd_avg_time, fuse_fwd_avg_time
        else:
            if split_fwd_bwd:
                torch_fwd_avg_time, torch_bwd_avg_time = self.test_single_speed_v2(self.torch_softmax, x_size, repeat_num)
                torch_avg_time = torch_fwd_avg_time + torch_bwd_avg_time
                fuse_fwd_avg_time, fuse_bwd_avg_time = self.test_single_speed_v2(self.fuse_softmax, x_size, repeat_num)
                fuse_avg_time = fuse_fwd_avg_time + fuse_bwd_avg_time

                fwd_acc_rate = (torch_fwd_avg_time - fuse_fwd_avg_time) / torch_fwd_avg_time * 100.0
                bwd_acc_rate = (torch_bwd_avg_time - fuse_bwd_avg_time) / torch_bwd_avg_time * 100.0
                fwd_bwd_acc_rate = (torch_avg_time - fuse_avg_time) / torch_avg_time * 100.0
                print('torch time: fwd={:.4f} ms, bwd ={:.4f} ms, fwd_bwd ={:.4f} ms\n'.
                      format(torch_fwd_avg_time, torch_bwd_avg_time, torch_avg_time),
                      'fuse time: fwd={:.4f} ms, bwd ={:.4f} ms, fwd_bwd ={:.4f} ms\n'.
                      format(fuse_fwd_avg_time, fuse_bwd_avg_time, fuse_avg_time),
                      "fwd accerate={:.2f}%, bwd accerate={:.2f}%, fwd_bwd accerate={:.2f}%".
                      format(fwd_acc_rate, bwd_acc_rate, fwd_bwd_acc_rate)
                      )
                print('=' * 30, '\n')
                return fwd_acc_rate, bwd_acc_rate
            else:
                torch_avg_time = self.test_single_speed(self.torch_softmax, x_size, False, repeat_num)
                fuse_avg_time = self.test_single_speed(self.fuse_softmax, x_size, False, repeat_num)
                fwd_bwd_acc_rate = (torch_avg_time - fuse_avg_time) / torch_avg_time * 100.0
                print('torch averange time={:.4f} ms\nfuse averange time={:.4f} ms, accelerate={:.2f}%'.format(
                    torch_avg_time, fuse_avg_time, fwd_bwd_acc_rate
                ))
                print('=' * 30, '\n')
                return torch_avg_time, fuse_avg_time


def test_lastdim_softmax():
    # 设置输入的softmax的tensor size
    x_size_lists = [
        (4, 256, 1024),
        (4, 256, 2048),
        (4, 256, 2049),
        (4, 256, 3966),
        (4, 256, 4096),
        (4, 256, 5015),
        (4, 256, 8192),
        (4, 16, 1025, 1025),
    ]

    # 设置测试输入half类型时，输出是否为float
    half_to_float = False
    # 设置是否测试half类型
    test_half = False

    for forward_only in [False, True]:  # 是否只测试推理
        for softmax_type in ['softmax', 'log_softmax']:
            # 设置测试类
            softmax_tester = SoftmaxTester(softmax_type=softmax_type, dim=-1,
                                           test_half=test_half, half_to_float=half_to_float)
            for x_size in x_size_lists:
                print('last dim test_half={}, forward_only={}, softmax_type={}, x_size={} '.
                      format(test_half, forward_only, softmax_type, x_size))
                softmax_tester.check_accuracy(x_size)
                softmax_tester.test_speed(x_size, forward_only=forward_only, split_fwd_bwd=True)


def test_spatial_softmax():
    # 设置输入的softmax的tensor size
    x_size_lists = [
        (4, 256, 1024),
        (4, 256, 2048),
        (4, 256, 2049),
        (4, 256, 3966),
        (4, 256, 4096),
        (4, 256, 5015),
        (4, 256, 8192),
        (4, 16, 1025, 1025),
    ]

    # 设置测试输入half类型时，输出是否为float
    half_to_float = False
    # 设置是否测试half类型
    test_half = False

    for forward_only in [False, True]:  # 是否只测试推理
        for softmax_type in ['softmax', 'log_softmax']:
            # 设置测试类
            softmax_tester = SoftmaxTester(softmax_type=softmax_type, dim=1,
                                           test_half=test_half, half_to_float=half_to_float)
            for x_size in x_size_lists:
                print('spatial dim test_half={}, forward_only={}, softmax_type={}, x_size={} '.
                      format(test_half, forward_only, softmax_type, x_size))
                softmax_tester.check_accuracy(x_size)
                softmax_tester.test_speed(x_size, forward_only=forward_only, split_fwd_bwd=True)


def main():
    test_lastdim_softmax()
    # test_spatial_softmax()


if __name__ == "__main__":
    main()
