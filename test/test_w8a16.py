import torch
import lightop
import time
from lightop.quant import fp8_to_16

torch.set_printoptions(precision=4,threshold=100000,linewidth=140,sci_mode=False)
torch.random.manual_seed(0)
repeat_num=100
acc=True
acc=False

def test(m,n,k,warps,nperblock,splitk):
    # type=torch.bfloat16
    type=torch.half
    a=torch.rand([m,k]).cuda().to(type)
    b=(torch.rand([n,k])-0.5).cuda().to(torch.float8_e4m3fn)
    scale=torch.rand([(n-1)//128+1,(k-1)//128+1]).cuda()
    bh = torch.empty(b.size(),dtype=type,device="cuda")
    # bh = (b.to(torch.float32) * scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:n, :k]).to(type)
    fp8_to_16(bh,b,scale)
    if not acc:
        for i in range(5):
            outref=torch.mm(a,bh.t())
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(repeat_num):
            outref=torch.mm(a,bh.t())
        torch.cuda.synchronize()
        end_time = time.time()
        fc1_espl = end_time - start_time

        for i in range(5):
            out=lightop.gemm_w8a16_fp8(a,b.t(),scale,warps,nperblock,splitk)
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(repeat_num):
            out=lightop.gemm_w8a16_fp8(a,b.t(),scale,warps,nperblock,splitk)
        torch.cuda.synchronize()
        end_time = time.time()
        fc2_espl = end_time - start_time

        print('m=',m,'n=',n,'k=',k,'warps=',warps,'nperblock=',nperblock,"splitk=",splitk, 'lightop time:', format(fc2_espl *1000/ repeat_num,'.3f'),'torch time:', format(fc1_espl *1000/ repeat_num,'.3f'))
    
    else:
        outref=torch.mm(a,bh.t())
        out=lightop.gemm_w8a16_fp8(a,b.t(),scale,warps,nperblock,splitk)
        print(out)
        print(outref)
        

    print(torch.allclose(out,outref,rtol=1e-02, atol=1e-02))


# m=[1,2,4,8,16]
m=[1,16]
# nk=[(1536,7168)]
# warps=[4,8,16]
# splitk=[1,2,4,8]
# nperblock=[16,32,64,128]

nk=[(1536,7168),(3072,1536),(576,7168),(4096,512),(7168,2048)]
warps=[0]
splitk=[0]
nperblock=[0]

if not acc:
    for M in m:
        for N,K in nk:
            for w in warps:
                for nper in nperblock:
                    for s in splitk:
                        test(M,N,K,w,nper,s)
else:
    test(1,3072,1536,0,0,0)
