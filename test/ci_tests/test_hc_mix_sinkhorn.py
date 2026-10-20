"""gfx928/gfx936/gfx938 native mHC: correctness, graph/stream contracts and complete-path timings.

python test/ci_tests/test_hc_mix_sinkhorn.py --quick --output /tmp/hc-quick
python test/ci_tests/test_hc_mix_sinkhorn.py --model /model/DeepSeek-V4.1-Flash-Channel-FP8 --output /tmp/hc-full
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import torch
from lightop.gemm_ops import (
    hc_mix_stats_sinkhorn_bf16x3 as BF,
    hc_mix_stats_sinkhorn_deepgemm as TF,
    split_bf16_hc_weight,
    split_tf32_hc_weight,
)

K=20480

def sink(mix,scale,base,iters=20,eps=1e-6):
    pre=torch.sigmoid(mix[:,:4]*scale[0]+base[:4])+eps
    post=2*torch.sigmoid(mix[:,4:8]*scale[1]+base[4:8])
    c=(mix[:,8:]*scale[2]+base[8:]).reshape(-1,4,4)
    c=torch.exp(c-c.amax(2,keepdim=True));c=c/c.sum(2,keepdim=True)+eps;c=c/(c.sum(1,keepdim=True)+eps)
    for _ in range(iters-1):c=c/(c.sum(2,keepdim=True)+eps);c=c/(c.sum(1,keepdim=True)+eps)
    return pre,post,c

def reference(x,w,s,b,iters=20,rms=1e-20,eps=1e-6):
    if not len(x):return sink(torch.empty(0,24,device=x.device),s,b,iters,eps)
    # Bound temporary memory for 65536-token inputs.
    rows=[]
    for chunk in x.split(512):
        f=chunk.float();rows.append((f@w.T)*torch.rsqrt(f.square().mean(-1,keepdim=True)+rms))
    return sink(torch.cat(rows),s,b,iters,eps)

def metric(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype and a.device==b.device
    d=(a.double()-b.double()).abs();bn=b.double().norm().item();dn=d.norm().item()
    v=int((d>1e-4+1e-4*b.abs()).sum())
    result=dict(max_abs=d.max().item() if d.numel() else 0,relative_l2=dn/bn if bn else (0 if not dn else float('inf')),violations=v,finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
    assert result['finite'] and not v,result
    return result

def compare(out,ref):return [metric(a,b) for a,b in zip(out,ref)]

def weights(model):
    torch.manual_seed(42)
    yield 'random',torch.randn(24,K)/math.sqrt(K),torch.tensor([.7,1.1,.5]),torch.randn(24)
    if model:
        from safetensors import safe_open
        index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
        for layer in (0,20,39):
            for kind in ('attn','ffn'):
                name=f'layers.{layer}.hc_{kind}_fn'
                with safe_open(model/index[name],framework='pt',device='cpu') as f:
                    yield name,f.get_tensor(name).float(),f.get_tensor(name.replace('_fn','_scale')).float(),f.get_tensor(name.replace('_fn','_base')).float()

def bf_supported(m):
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    return m <= 131072 if arch == "gfx938" else m == 0 or 4096 <= m <= 65536

@torch.inference_mode()
def correctness(args):
    records=[]
    ms=[0,1,6,12,17,32,33,64,65,128,129,256,257,4096] if args.quick else [0,1,2,6,10,12,15,16,17,31,32,33,42,48,49,63,64,65,127,128,129,191,192,255,256,257,511,512,513,1024,2048,4095,4096,4097]
    for name,wc,sc,bc in weights(args.model):
        w,s,b=[v.cuda() for v in (wc,sc,bc)]
        tp=split_tf32_hc_weight(w);bp=split_bf16_hc_weight(w)
        assert torch.equal(tp[0].view(torch.int32),w.view(torch.int32)&-8192)
        assert torch.equal(tp[1],w-tp[0])
        assert torch.equal(bp[0],w.bfloat16())
        for seed in range(1 if args.quick else 3):
            torch.manual_seed(20261017+seed)
            raw=torch.randn(max(ms),K,device='cuda',dtype=torch.bfloat16)
            for pattern in (['normal'] if args.quick else ['normal','tiny','wide','zero','cancel']):
                if pattern=='normal':x=raw
                elif pattern=='tiny':x=(raw.float()*1e-7).bfloat16()
                elif pattern=='wide':x=(raw.float()*torch.exp2(torch.linspace(-12,12,K,device='cuda'))).bfloat16()
                elif pattern=='zero':x=torch.zeros_like(raw)
                else:
                    x=raw.clone();x[:,1::2]=-x[:,::2]
                for m in ms:
                    ref=reference(x[:m],w,s,b)
                    for label,fn,parts in [('tf32',TF,tp)]+([('bf16x3',BF,bp)] if bf_supported(m) else []):
                        out=fn(x[:m],parts,s,b,20,1e-20,1e-6)
                        records.append(dict(weight=name,seed=seed,pattern=pattern,m=m,backend=label,metrics=compare(out,ref)))
            print('ACCURACY',name,seed,'cases',len(records),flush=True)
        # Independent FP64 reference on a small same-input matrix.
        xd=raw[:17].cpu().double();wd=wc.double()
        mix=(xd@wd.T)*torch.rsqrt(xd.square().mean(-1,keepdim=True)+1e-20)
        ref64=tuple(t.float().cuda() for t in sink(mix,sc.double(),bc.double()))
        records.append(dict(weight=name,backend='tf32-vs-fp64',m=17,metrics=compare(TF(raw[:17],tp,s,b,20,1e-20,1e-6),ref64)))
        (args.output/'accuracy.json').write_text(json.dumps(records,indent=2))
    if not args.quick:
        for m in [65535,65536,65537,131071,131072]:
            x=torch.randn(m,K,device='cuda',dtype=torch.bfloat16)
            ref=reference(x,w,s,b)
            for label,fn,parts in [('tf32',TF,tp)]+([('bf16x3',BF,bp)] if bf_supported(m) else []):
                records.append(dict(weight=name,m=m,backend=label,metrics=compare(fn(x,parts,s,b,20,1e-20,1e-6),ref)))
            print('BOUNDARY',m,flush=True)
        (args.output/'accuracy.json').write_text(json.dumps(records,indent=2))
    return len(records)

@torch.inference_mode()
def contracts(args):
    records=[]
    for device in range(min(torch.cuda.device_count(),2)):
        with torch.cuda.device(device):
            for fn,split,m in [(TF,split_tf32_hc_weight,17)]+[(BF,split_bf16_hc_weight,m) for m in (17,42,65,192,4096) if bf_supported(m)]:
                x=torch.randn(m,K,device='cuda',dtype=torch.bfloat16);w=torch.randn(24,K,device='cuda')*.01
                s=torch.ones(3,device='cuda');b=torch.zeros(24,device='cuda');parts=split(w)
                initial=x.clone();wp=[p.clone() for p in parts]
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):fn(x,parts,s,b,20,1e-20,1e-6)
                torch.cuda.current_stream().wait_stream(stream)
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,stream=stream):out=fn(x,parts,s,b,20,1e-20,1e-6)
                g.replay();compare(out,reference(x,w,s,b))
                assert torch.equal(x,initial) and all(torch.equal(a,b) for a,b in zip(parts,wp))
                x[:,::2].neg_();w.mul_(1.5)
                for dst,src in zip(parts,split(w)):dst.copy_(src)
                g.replay();compare(out,reference(x,w,s,b))
                # Unaligned contiguous storage-offset views.
                xx=torch.empty(x.numel()+1,device=x.device,dtype=x.dtype)[1:].view_as(x);xx.copy_(x)
                compare(fn(xx,parts,s,b,20,1e-20,1e-6),reference(x,w,s,b))
                records.append(dict(device=device,backend=fn.__name__,graph_update=True,input_unchanged=True,stream=True,unaligned=True))
    # Invalid metadata must fail before a kernel launch.
    with torch.cuda.device(0):
        x=torch.zeros(1,K,device='cuda',dtype=torch.bfloat16);w=torch.zeros(24,K,device='cuda');s=torch.ones(3,device='cuda');b=torch.zeros(24,device='cuda')
        for fn,split in [(TF,split_tf32_hc_weight),(BF,split_bf16_hc_weight)]:
            p=split(w)
            bad=[(x.float(),p,s,b,20,1e-20,1e-6),(x,p[:-1],s,b,20,1e-20,1e-6),(x,p,s[:2],b,20,1e-20,1e-6),(x,p,s,b[:23],20,1e-20,1e-6),(x,p,s,b,0,1e-6,1e-6),(x,p,s,b,20,0,1e-6),(x,p,s,b,20,1e-6,float('nan')),(x[:,::2],p,s,b,20,1e-20,1e-6),(x.cpu(),p,s,b,20,1e-20,1e-6),(x,p,s,b,20,1e-20,1e-6,3)]
            for a in bad:
                try:fn(*a)
                except (RuntimeError,ValueError):pass
                else:raise AssertionError('Invalid input accepted')
            records.append(dict(backend=fn.__name__,invalid_cases=len(bad)))
        if bf_supported(1):
            outside=torch.empty(131073,K,device='cuda',dtype=torch.bfloat16)
            try:BF(outside,split_bf16_hc_weight(w),s,b,20,1e-20,1e-6)
            except (RuntimeError,ValueError):pass
            else:raise AssertionError('BF16x3 accepted an unvalidated M')
            records.append(dict(backend='bf16x3',out_of_range_rejected=True))
    (args.output/'contracts.json').write_text(json.dumps(records,indent=2))

def timings(fn,graph=False):
    for _ in range(10):fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();y=fn();torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()-before;del y
    count=20 if graph else 1
    if graph:
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(count):fn()
        call=g.replay
        for _ in range(5):call()
    else:call=fn
    samples=[]
    for _ in range(10):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();call();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/count)
    return dict(mean_us=statistics.mean(samples),p50_us=statistics.median(samples),p90_us=sorted(samples)[8],cv=statistics.pstdev(samples)/statistics.mean(samples),peak_bytes=peak,samples_us=samples)

@torch.inference_mode()
def benchmark(args):
    from sglang.kernels.ops.layernorm.mhc import hc_split_sinkhorn
    data=list(weights(args.model));name,w,s,b=data[min(1,len(data)-1)];w,s,b=[v.cuda() for v in (w,s,b)]
    tp=split_tf32_hc_weight(w);bp=split_bf16_hc_weight(w)
    rows=[]
    def finish(mix):return tuple(v.squeeze(1) for v in hc_split_sinkhorn(mix.unsqueeze(1),s,b,4,20,1e-6))
    measured_ms = getattr(args, 'm_values', None) or ([1,17,128,4096] if args.quick else [1,2,10,12,16,17,32,48,64,128,256,512,1024,2048,4096,8192,16384,32768,65536])
    for m in measured_ms:
        x=torch.randn(m,K,device='cuda',dtype=torch.bfloat16)
        def torch_mix():
            f=x.float();return finish((f@w.T)*torch.rsqrt(f.square().mean(-1,keepdim=True)+1e-20))
        funcs={'torch':torch_mix,'tf32':lambda:TF(x,tp,s,b,20,1e-20,1e-6)}
        if bf_supported(m):funcs['bf16x3']=lambda:BF(x,bp,s,b,20,1e-20,1e-6)
        ref=reference(x,w,s,b)
        for label,fn in funcs.items():
            compare(fn(),ref)
            for graph in [False,True]:
                result=dict(m=m,backend=label,graph=graph,**timings(fn,graph));rows.append(result)
                print('BENCH',m,label,'graph',graph,round(result['mean_us'],3),flush=True)
        (args.output/'performance.json').write_text(json.dumps(rows,indent=2))
    split={}
    for label,fn in [('tf32',split_tf32_hc_weight),('bf16x3',split_bf16_hc_weight)]:
        split[label]=dict(**timings(lambda:fn(w)),cache_bytes=sum(p.numel()*p.element_size() for p in fn(w)))
    (args.output/'split.json').write_text(json.dumps(split,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--model',type=Path);p.add_argument('--output',type=Path,default=Path('hc_mix_sinkhorn_results'),help='Result directory (default: ./hc_mix_sinkhorn_results)');p.add_argument('--bench-only',action='store_true');p.add_argument('--accuracy-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8)
    if not a.bench_only:correctness(a);contracts(a)
    if not a.accuracy_only:benchmark(a)
    print('PASS',flush=True)
def test_compensated_smoke(tmp_path):
    import pytest
    from types import SimpleNamespace
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] not in ("gfx928", "gfx936", "gfx938"):
        pytest.skip("native compensated mHC requires gfx928, gfx936 or gfx938")
    torch.backends.cuda.matmul.allow_tf32 = False
    correctness(SimpleNamespace(quick=True, model=None, output=tmp_path))


def test_compensated_graph_contracts(tmp_path):
    import pytest
    from types import SimpleNamespace
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] not in ("gfx928", "gfx936", "gfx938"):
        pytest.skip("native compensated mHC requires gfx928, gfx936 or gfx938")
    contracts(SimpleNamespace(output=tmp_path))


if __name__=='__main__':main()
