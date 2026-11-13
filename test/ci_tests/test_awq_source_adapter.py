"""Exercise production AWQ entry points without either precompiled AWQ archive."""
import subprocess
import torch
import pytest
from lightop import op

def _case(m, n, k, group_size, storage_k=None, device="cuda", seed=20261030):
    storage_k = storage_k or k
    rng = torch.Generator().manual_seed(seed + m + n + k)
    q = torch.randint(0, 16, (n, storage_k), generator=rng, dtype=torch.int64)
    z = torch.randint(0, 16, (n, storage_k // group_size), generator=rng, dtype=torch.int64)
    scale = (torch.rand(z.shape, generator=rng) * 0.45 + 0.03).half()
    positions = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int64)
    packed = (q.reshape(n, storage_k // 8, 8) << (4 * positions)).sum(-1).int()
    scale_zero = torch.stack(((64 + z).half(), scale), dim=-1).contiguous().view(torch.int32).reshape(n, -1)
    logical = ((q.float() - z.repeat_interleave(group_size, 1).float()) * scale.repeat_interleave(group_size, 1).float()).half()[:, :k]
    a = (torch.randn(m, k, generator=rng) / 8).half()
    return a.to(device), packed.to(device), scale_zero.to(device), logical, (a.float() @ logical.float().T).half()



def _gemm(a, packed, sz, m, n, k, storage_k, group):
    # These legacy arguments remain accepted, but the source implementation
    # allocates its own temporaries and does not consume the vendor workspace.
    workspace = torch.empty(0, device=a.device, dtype=a.dtype)
    return op._lmslimquant_gemm_w4a16_v2(
        a, packed, sz, m, n, k, a.shape[1], storage_k, storage_k,
        n, group, workspace, 0, "W2Atom1_M32N32K32L1_Bf2_Uf0_Ul1_Gv0")


@pytest.mark.parametrize('m,n,k,g,storage_k', [
    (1,128,128,128,128), (17,128,384,128,384), (65,128,512,128,512),
    (16,128,256,32,256), (16,128,256,64,256), (16,128,128,8,128),
    (17,128,384,128,512), (16,128,192,64,256), (16,120,256,128,256),
    (3,17,128,128,128), (512,128,128,128,128), (513,128,128,128,256),
])
def test_production_legacy_interfaces(m,n,k,g,storage_k):
    assert torch.cuda.is_available() and torch.version.hip
    a,p,sz,w,expected = _case(m,n,k,g,storage_k)
    storage = torch.full((m*storage_k+16,), 123, device=a.device, dtype=a.dtype)
    op._lmslimquant_awq_input_padding(a,m,k,storage_k,storage,m*storage_k*2)
    padded = storage[:m*storage_k].view(m,storage_k)
    actual = _gemm(padded,p,sz,m,n,k,storage_k,g)
    dq = torch.full((n*k+16,),123,device=a.device,dtype=a.dtype)
    op._lmslimquant_awq_dequant_colmajor(dq,p,sz,k,storage_k,n,g,n*k*2)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu(),expected,atol=1e-2,rtol=1e-3)
    torch.testing.assert_close(dq[:n*k].view(n,k).cpu(),w,atol=0,rtol=0)
    torch.testing.assert_close(padded[:,:k],a,atol=0,rtol=0)
    assert torch.count_nonzero(padded[:,k:]) == 0
    assert bool((storage[m*storage_k:]==123).all())
    assert bool((dq[n*k:]==123).all())


@pytest.mark.parametrize('k,g',[(128,128),(256,32),(384,128),(512,64),(192,64)])
def test_column_major_dequant(k,g):
    _,p,sz,w,_=_case(1,120,k,g)
    out=torch.empty((120,k),device=p.device,dtype=torch.float16)
    op.gemm_awq_w4a16_dequant(out,p,sz,False)
    torch.testing.assert_close(out.cpu(),w,atol=0,rtol=0)


@pytest.mark.parametrize('k,g',[(128,128),(256,32),(384,128),(512,64)])
def test_marlin_dequant(k,g):
    _,p,sz,w,_=_case(1,120,k,g)
    tile=256 if k%256==0 else 128
    repacked=p.view(120,k//tile,tile//8).permute(1,0,2).contiguous().view(k//tile,120*tile//8)
    out=torch.empty((120,k),device=p.device,dtype=torch.float16)
    op.gemm_awq_w4a16_dequant(out,repacked,sz,True)
    torch.testing.assert_close(out.cpu(),w,atol=0,rtol=0)


@pytest.mark.parametrize('m,n,nw,kw,np,kp,split,marlin,g',[
    (1,128,1,1,16,128,2,False,128),
    (13,128,1,2,32,128,4,False,128),
    (17,128,2,2,32,256,3,True,128),
    (32,128,1,4,64,128,4,False,64),
    (17,128,1,8,16,128,2,False,128),
    (17,128,1,16,16,128,2,False,128),
    (17,120,2,2,32,128,4,False,128),
    (17,120,1,1,16,128,2,True,32),
    (17,128,1,1,16,256,4,True,32),
    (17,128,1,2,16,128,3,False,256),
    (17,128,1,2,16,128,3,True,1024),
    (13,128,1,2,128,256,1,True,64),
    (13,128,1,2,128,256,100,False,128),
])
def test_splitk_and_tail_fallback(m,n,nw,kw,np,kp,split,marlin,g):
    for seed in [20261030,20261031]:
        a,p,sz,_,expected=_case(m,n,1024,g,seed=seed)
        if marlin:
            p=p.view(n,1024//kp,kp//8).permute(1,0,2).contiguous().view(1024//kp,n*kp//8)
        actual=torch.full((m,n),float('nan'),device=a.device,dtype=a.dtype)
        op.gemm_awq_w4a16(actual,a,p,sz,nw,kw,np,kp,split,marlin)
        torch.testing.assert_close(actual.cpu(),expected,atol=1e-2,rtol=1e-3)


@pytest.mark.parametrize('m,k,g', [(17,256,128), (65,384,128)])
def test_format_to_production_gemm(m,k,g):
    # Standard AWQ packs N with this interleaving. Exercise the real format and
    # transpose operators before the replacement GEMM, with nonconstant values.
    n=128
    rng=torch.Generator().manual_seed(20261031)
    q=torch.randint(0,16,(k,n),generator=rng,dtype=torch.int64)
    z=torch.randint(0,16,(k//g,n),generator=rng,dtype=torch.int64)
    scales=(torch.rand(z.shape,generator=rng)*0.45+0.03).half()
    shifts=4*torch.tensor([0,4,1,5,2,6,3,7],dtype=torch.int64)
    def pack(values):
        return (values.reshape(values.shape[0],n//8,8)<<shifts).sum(-1).int().cuda()
    qweight,qzeros=pack(q),pack(z)
    converted=torch.zeros_like(qweight)
    sz=torch.empty((k//g,n),device='cuda',dtype=torch.int32)
    workspace=torch.empty_like(scales,device='cuda')
    op._lmslimquant_convert_s4_k_m8(converted,sz,workspace,qweight,scales.cuda(),qzeros,n,k,g)
    transposed=torch.empty_like(sz)
    op._lmslimquant_permute_trans(transposed,sz,k//g,n)
    a=(torch.randn(m,k,generator=rng)/8).half()
    logical=((q.float()-z.repeat_interleave(g,0).float())*scales.repeat_interleave(g,0).float()).half()
    expected=(a.float()@logical.float()).half()
    actual=_gemm(a.cuda(),converted,transposed,m,n,k,k,g)
    torch.testing.assert_close(actual.cpu(),expected,atol=1e-2,rtol=1e-3)


def test_two_streams_and_repeated_splitk():
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    cases=[_case(17,128,1024,128,seed=20261030+i) for i in range(2)]
    outputs=[]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for repeat in range(8):
        for stream,(a,p,sz,_,expected) in zip(streams,cases):
            with torch.cuda.stream(stream):
                actual=torch.empty((17,128),device=a.device,dtype=a.dtype)
                op.gemm_awq_w4a16(actual,a,p,sz,1,2,16,128,4,False)
                compat=_gemm(a,p,sz,17,128,1024,1024,128)
                outputs.extend([(actual,expected),(compat,expected)])
    for stream in streams: stream.synchronize()
    for actual,expected in outputs:
        torch.testing.assert_close(actual.cpu(),expected,atol=1e-2,rtol=1e-3)


@pytest.mark.skipif(torch.cuda.device_count()<2,reason="requires two GPUs")
def test_device_guard_and_workspace_isolation():
    original=torch.cuda.current_device()
    try:
        for device in [1,0,1]:
            a,p,sz,w,expected=_case(17,128,1024,128,device=f"cuda:{device}")
            actual=torch.empty((17,128),device=a.device,dtype=a.dtype)
            torch.cuda.set_device(1-device)
            op.gemm_awq_w4a16(actual,a,p,sz,1,2,16,128,4,False)
            assert torch.cuda.current_device()==1-device
            compat=_gemm(a,p,sz,17,128,1024,1024,128)
            dq=torch.empty((128,1024),device=a.device,dtype=a.dtype)
            op._lmslimquant_awq_dequant_colmajor(dq,p,sz,1024,1024,128,128,dq.numel()*2)
            assert torch.cuda.current_device()==1-device
            torch.testing.assert_close(actual.cpu(),expected,atol=1e-2,rtol=1e-3)
            torch.testing.assert_close(compat.cpu(),expected,atol=1e-2,rtol=1e-3)
            torch.testing.assert_close(dq.cpu(),w,atol=0,rtol=0)
    finally:
        torch.cuda.set_device(original)


def test_rejects_bad_dtype_capacity_and_alias():
    a,p,sz,_,_=_case(16,128,128,128)
    with pytest.raises(RuntimeError,match="dtype"):
        _gemm(a.bfloat16(),p,sz,16,128,128,128,128)
    with pytest.raises(RuntimeError,match="insufficient"):
        _gemm(a,p[:,:-1].contiguous(),sz,16,128,128,128,128)
    with pytest.raises(RuntimeError,match="workspace_size"):
        op._lmslimquant_awq_input_padding(a,16,128,256,torch.empty((16,256),device=a.device,dtype=a.dtype),1)
    with pytest.raises(RuntimeError):
        op._lmslimquant_awq_input_padding(a,16,128,128,a,a.numel()*2)
    with pytest.raises(RuntimeError,match="workspace_size"):
        op._lmslimquant_awq_dequant_colmajor(torch.empty((128,128),device=a.device,dtype=a.dtype),p,sz,128,128,128,128,1)


@pytest.mark.parametrize('m,n,k',[(0,128,128),(1,0,128),(3,16,0)])
def test_empty_gemm(m,n,k):
    a=torch.empty((m,k),device='cuda',dtype=torch.float16)
    p=torch.empty((n,k//8),device='cuda',dtype=torch.int32)
    sz=torch.empty((n,k//128),device='cuda',dtype=torch.int32)
    actual=_gemm(a,p,sz,m,n,k,k,128)
    assert actual.shape==(m,n)
    if k==0: assert torch.count_nonzero(actual)==0


def test_no_legacy_awq_archive_symbols():
    output=subprocess.check_output(['readelf','--dyn-syms','--wide',op.__file__],text=True)
    for symbol in ['_Z10gemm_w4a16PKv','_Z13input_paddingPKv',
                   '_Z24dequant_w4_gemm_colmajorPKv','w4a16_awq_cute_namespace']:
        assert symbol not in output, symbol
