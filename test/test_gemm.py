import torch
import lightop
import time

acc=True
acc=False

def test(m,n,k):

    torch.manual_seed(0)
    a=torch.rand([m,k]).cuda().half()
    scale=torch.rand(n).cuda().half()*0.1
    zero=torch.rand(n).cuda().half()
    b_w4=torch.randint(4294967296,[n,k//8]).cuda().to(torch.uint32)
    b=lightop.w4tow16(b_w4,scale,zero)

    if acc:
        repeat_num = 1
        print(a)
        print(b)
    else :
        repeat_num = 100

    if not acc:
        for i in range(5):
            d1=torch.mm(a,b.t())

    torch.cuda.synchronize()
    start_time = time.time()

    for i in range(repeat_num):
        d1=torch.mm(a,b.t())

    torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time

    if not acc:
        for i in range(5):
            d2=lightop.gemm_w4a16(a,b_w4.t(),scale,zero)

    torch.cuda.synchronize()
    start_time = time.time()

    for i in range(repeat_num):
        d2=lightop.gemm_w4a16(a,b_w4.t(),scale,zero)

    torch.cuda.synchronize()
    end_time = time.time()
    fc2_espl = end_time - start_time
    print('m=',m,'n=',n,'k=',k,' roblass time:', format(fc1_espl *1000/ repeat_num,'.3f'), 'ms. lightop time:', format(fc2_espl *1000/ repeat_num,'.3f'))

    check = torch.allclose(d1, d2, rtol=1e-2, atol=1e-2)
    if check:
        print("test pass")
    else :
        print(d1)
        print(d2)

if not acc:
    M=[1,2,4,8,16]
    # M=[1,2,4]
    NK=[(3328,6144),(6144,3072),(24576,6144),(6144,12288)]

    for n,k in NK:
        for m in M:
            test(m,n,k)
else :
    test(1,32,128)
