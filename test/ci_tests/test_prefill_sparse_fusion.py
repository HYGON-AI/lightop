# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Independent bitwise references for sparse Prefill pointwise fusions."""

import argparse, importlib.util, json, pathlib, sys
import torch


def reference_query(q, raw, freqs, positions, rope, scale, ratio, width):
    rows, heads, _ = q.shape
    values = q.float()
    real = values[..., 0::2]
    imag = values[..., 1::2]
    if rope:
        f = torch.view_as_real(freqs[positions.long()])
        cosine = f[:, None, :, 0]
        sine = f[:, None, :, 1]
        vr = real[..., -rope // 2 :]
        vi = imag[..., -rope // 2 :]
        rr = (vr * cosine - vi * sine).bfloat16().float()
        ii = (vr * sine + vi * cosine).bfloat16().float()
        values = values.clone()
        values[..., -rope::2] = rr
        values[..., -rope + 1 :: 2] = ii
    groups = values.reshape(rows, heads, 4, 32)
    bits = (
        (groups.abs().amax(-1).clamp_min(6 * 2.0**-126) * (1.0 / 6.0))
        .contiguous()
        .view(torch.int32)
    )
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
    factor = (exponent << 23).view(torch.float32)
    scaled = (groups / factor[..., None]).clamp(-6, 6)
    mag = scaled.abs()
    step = torch.where(mag < 2, 0.5, torch.where(mag < 4, 1.0, 2.0))
    rounded = (torch.round(mag / step) * step * scaled.sign()).reshape(rows, heads, 128)
    code = torch.zeros_like(rounded, dtype=torch.uint8)
    for threshold in [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]:
        code += (rounded.abs() > threshold).to(torch.uint8)
    code |= ((rounded < 0) & (code != 0)).to(torch.uint8) << 3
    payload = (code[..., 0::2] | (code[..., 1::2] << 4)).view(torch.int8)[:, None]
    shifts = torch.arange(4, device=q.device, dtype=torch.int64) * 8
    sf = (exponent.long() << shifts).sum(-1).int()[:, None]
    weights = (raw * scale).float()
    lens = ((positions + 1) // ratio).int()
    kept = torch.minimum((lens + 7) // 8, torch.full_like(lens, width))
    valid = torch.where(lens > 0, 8 * (kept - 1) + (lens - 1) % 8 + 1, 0)
    return (
        payload,
        sf,
        weights,
        lens,
        valid,
        torch.arange(rows, device=q.device, dtype=torch.int32) // 2,
        torch.zeros_like(lens),
    )


def run_tests(m, quick=False):
    torch.manual_seed(471319)
    records = []
    cases = [
        (65, 32, 64, 1, 1.0),
        (9, 4, 0, 4, 0.0),
        (13, 12, 128, 2, 1.0e-20),
        (1, 8, 64, 1, 1.0e10),
    ]
    if not quick:
        cases += [
            (4095, 32, 64, 1, 1.0),
            (64, 16, 32, 4, 1.0),
            (0, 32, 64, 1, 1.0),
            (3, 20, 96, 1, 1.0),
        ]
    for rows, heads, rope, ratio, magnitude in cases:
        q = (
            torch.randn((rows, heads, 128), device="cuda", dtype=torch.bfloat16)
            * magnitude
        )
        raw = torch.randn((rows, heads), device="cuda", dtype=torch.bfloat16)
        pos = torch.randint(0, 128, (rows,), device="cuda", dtype=torch.int32)
        angles = torch.randn((128, rope // 2), device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        scale = 128**-0.5 * heads**-0.5
        actual = m.prepare_prefill_sparse_query(
            q, raw, freqs, pos, rope, scale, ratio, 2048
        )
        expected = reference_query(q, raw, freqs, pos, rope, scale, ratio, 2048)
        for name, a, b in zip(
            ["payload", "scale", "weights", "lengths", "valid", "pairs", "offsets"],
            actual,
            expected,
        ):
            assert torch.equal(a, b), (name, rows, heads, rope, (a != b).sum().item())
        records.append(
            {
                "kind": "prepare_query",
                "rows": rows,
                "heads": heads,
                "rope": rope,
                "ratio": ratio,
                "status": "bitwise PASS",
            }
        )
        print(json.dumps(records[-1]), flush=True)
    for rows, width, sw, topk, ratio in [
        (65, 2048, 2052, 512, 1),
        (7, 16, 12, 512, 4),
        (1, 12, 16, 1024, 2),
    ]:
        context = 4096
        page = 64
        map = torch.arange(context * ratio, device="cuda", dtype=torch.int32)[
            None, :
        ].repeat(3, 1)
        map += torch.arange(3, device="cuda", dtype=torch.int32)[:, None] * (
            context * ratio
        )
        request = torch.tensor([2], device="cuda", dtype=torch.int32)
        source = torch.randint(
            -2, context // 8, (rows, sw), device="cuda", dtype=torch.int32
        )
        blocks, pages = m.prepare_prefill_sparse_candidates(
            source, width, map, request, context, ratio, page
        )
        b = torch.full((rows, width), 2147483647, device="cuda", dtype=torch.int32)
        n = min(width, sw)
        b[:, :n] = torch.where(source[:, :n] >= 0, source[:, :n], 2147483647)
        b = b.sort(-1).values
        p = (
            map[
                request.long(), torch.arange(context // 64, device="cuda") * 64 * ratio
            ].long()
            // ratio
            // 64
        ).int()
        assert torch.equal(blocks, b) and torch.equal(pages, p)
        lengths = torch.randint(1, context, (rows,), device="cuda", dtype=torch.int32)
        lengths[0] = 0
        valid = torch.minimum((lengths + 7) // 8, torch.full_like(lengths, width))
        valid = 8 * (valid - 1) + (lengths - 1) % 8 + 1
        compact = torch.randint(
            -2, width * 8 + 5, (rows, topk), device="cuda", dtype=torch.int32
        )
        out = torch.empty((rows, topk + 7), device="cuda", dtype=torch.int32)[:, :topk]
        out_pages = torch.empty_like(out)
        got = m.finalize_prefill_sparse_indices(
            compact,
            blocks,
            lengths,
            valid,
            out_logical=out,
            req_to_token=map,
            request_id=request,
            compress_ratio=ratio,
            out_page_indices=out_pages,
        )
        live = (compact >= 0) & (compact < valid[:, None])
        slots = (compact.clamp_min(0) // 8).clamp_max(width - 1)
        logical = blocks.gather(1, slots) * 8 + compact.clamp_min(0) % 8
        live &= (logical >= 0) & (logical < lengths[:, None])
        logical = torch.where(live, logical, 2147483647).sort(-1).values
        ref = torch.where(logical != 2147483647, logical, -1)
        physical = (
            map[request.long(), ref.clamp_min(0).long() * ratio].long() // ratio
        ).int()
        physical = torch.where(ref >= 0, physical, -1)
        assert torch.equal(got, ref) and torch.equal(out_pages, physical)
        records.append(
            {
                "kind": "candidate_and_finalize",
                "rows": rows,
                "width": width,
                "input_width": sw,
                "topk": topk,
                "ratio": ratio,
                "status": "bitwise PASS",
            }
        )
        print(json.dumps(records[-1]), flush=True)
    # Exercise the adaptive ordered branch, including invalid IDs at the tail.
    source = torch.arange(2048, device="cuda", dtype=torch.int32)[None, :].repeat(65, 1)
    source[:, -8:] = -1
    mapping = torch.arange(65536, device="cuda", dtype=torch.int32)[None, :]
    request = torch.zeros(1, device="cuda", dtype=torch.int32)
    blocks, pages = m.prepare_prefill_sparse_candidates(
        source, 2048, mapping, request, 65536, 1, 64
    )
    expected = source.clone()
    expected[expected < 0] = 2147483647
    assert torch.equal(blocks, expected)
    compact = torch.arange(512, device="cuda", dtype=torch.int32)[None, :].repeat(65, 1)
    lengths = torch.full((65,), 65536, device="cuda", dtype=torch.int32)
    valid = torch.full_like(lengths, 16384)
    physical = torch.empty_like(compact)
    logical = m.finalize_prefill_sparse_indices(
        compact,
        blocks,
        lengths,
        valid,
        req_to_token=mapping,
        request_id=request,
        out_page_indices=physical,
    )
    assert torch.equal(logical, compact) and torch.equal(physical, compact)
    records.append({"kind": "already-ordered-with-padding", "status": "bitwise PASS"})
    print(json.dumps(records[-1]), flush=True)
    if hasattr(m, "_prepare_prefill_sparse_request"):
        q = torch.randn((65, 32, 128), device="cuda", dtype=torch.bfloat16)
        raw = torch.randn((65, 32), device="cuda", dtype=torch.bfloat16)
        pos = torch.arange(65, device="cuda", dtype=torch.int32)
        angles = torch.randn((128, 32), device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        expected = reference_query(q, raw, freqs, pos, 64, 1.0 / 64, 1, 2048) + (
            blocks,
            pages,
        )
        native_flags = (
            torch.empty((q.shape[0],), device="cuda", dtype=torch.int32)
            if hasattr(m, "_index_sort_backend") and m._index_sort_backend() is not None
            else None
        )
        actual = m._prepare_prefill_sparse_request(
            q,
            raw,
            freqs,
            pos,
            64,
            1.0 / 64,
            1,
            2048,
            source,
            mapping,
            request,
            65536,
            64,
            **({"row_flags": native_flags} if native_flags is not None else {}),
        )
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert torch.equal(a, b), ("combined-preparation", i)
        records.append({"kind": "combined-query-geometry", "status": "bitwise PASS"})
        print(json.dumps(records[-1]), flush=True)
    if hasattr(m, "_index_sort_backend") and m._index_sort_backend() is not None:
        # GPU-produced certificates, duplicate-preserving fallback, full-domain
        # keys, integer map dtypes/floor division and dynamic Graph decisions.
        for domain, ratio in [(32768, 1), (32768, 2), (65536, 3), (1048576, 4)]:
            rows, width, k = 9, 2048, 512
            source = (
                torch.rand((rows, 4096), device="cuda")
                .argsort(-1)[:, :width]
                .int()
                .contiguous()
            )
            source[1, 1] = source[1, 0]
            source[2, :].fill_(-1)
            source[3, :] = source[3, :].sort().values
            if domain > 65536:
                source[4, 0] = 100000
            lengths = torch.full((rows,), domain, device="cuda", dtype=torch.int32)
            valid = torch.full_like(lengths, width * 8)
            flags = torch.empty_like(lengths)
            mapping = (
                torch.arange(domain * ratio, device="cuda", dtype=torch.int64)
                - domain * ratio
                - 7
            )[None, :]
            request = torch.zeros(1, device="cuda", dtype=torch.int64)
            blocks, pages = m.prepare_prefill_sparse_candidates(
                source, width, mapping, request, domain, ratio, 64, row_flags=flags
            )
            expected_pages = (
                mapping[
                    0, torch.arange((domain + 63) // 64, device="cuda") * 64 * ratio
                ]
                // ratio
                // 64
            ).int()
            assert torch.equal(pages, expected_pages)
            refb = torch.where(source >= 0, source, 2147483647).sort(-1).values
            assert torch.equal(blocks, refb)
            unique = ((refb[:, 1:] != refb[:, :-1]) | (refb[:, 1:] == 2147483647)).all(
                -1
            )
            # Radix conservatively emits zero; a one must always be valid.
            assert bool(((flags != 1) | unique).all())
            compact = (
                torch.rand((rows, width * 8), device="cuda")
                .argsort(-1)[:, :k]
                .int()
                .contiguous()
            )
            compact[5, 1] = compact[5, 0]
            compact[6, :].fill_(-1)
            logical = torch.empty((rows, k + 3), device="cuda", dtype=torch.int32)[
                :, :k
            ]
            physical = torch.empty_like(logical)

            def run():
                m.finalize_prefill_sparse_indices(
                    compact,
                    blocks,
                    lengths,
                    valid,
                    out_logical=logical,
                    req_to_token=mapping,
                    request_id=request,
                    compress_ratio=ratio,
                    out_page_indices=physical,
                    candidate_flags=flags,
                )

            def check():
                slots = (compact.clamp_min(0) // 8).clamp_max(width - 1)
                pos = blocks.gather(1, slots).long() * 8 + compact.clamp_min(0) % 8
                good = (
                    (compact >= 0)
                    & (compact < valid[:, None])
                    & (pos >= 0)
                    & (pos < lengths[:, None])
                )
                pos = torch.where(good, pos, 2147483647).sort(-1).values
                ref = torch.where(pos == 2147483647, -1, pos).int()
                phys = (
                    (mapping[0, ref.clamp_min(0).long() * ratio] // ratio)
                    .int()
                    .masked_fill(ref < 0, -1)
                )
                assert torch.equal(logical, ref) and torch.equal(physical, phys), domain

            run()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            compact.copy_(compact.flip(1))
            graph.replay()
            torch.cuda.synchronize()
            check()
            records.append(
                {
                    "kind": "native-sort-contracts",
                    "domain": domain,
                    "status": "bitwise PASS",
                }
            )
            print(json.dumps(records[-1]), flush=True)
    records.extend(run_short_row_tests(m))
    return records



def run_short_row_tests(m):
    records=[]
    backend=m._native_prefill_fusion_backend()
    # R16 is the CP8 bounded-replay shape; exercise the fused-sort branch,
    # including the two-CTA query layout and both integer pointer widths.
    for rows,lanes,wide in [(1,4,False),(16,8,False),(17,16,True),(32,8,True)]:
        context=32768
        q=torch.randn((rows,32,128),device="cuda",dtype=torch.bfloat16)
        raw=torch.randn((rows,32),device="cuda",dtype=torch.bfloat16)
        pos=torch.arange(rows,device="cuda",dtype=torch.int64 if wide else torch.int32)+9
        angles=torch.randn((64,32),device="cuda");freq=torch.polar(torch.ones_like(angles),angles)
        mapping=torch.arange(context,device="cuda",dtype=torch.int64 if wide else torch.int32)[None,:]
        request=torch.zeros(1,device="cuda",dtype=pos.dtype)
        source=torch.arange(2048,device="cuda",dtype=torch.int32)[None,:].repeat(rows,1)
        flags=torch.empty((rows,),device="cuda",dtype=torch.int32)
        def run():
            return backend.prefill_sparse_native_prepare_query(q,raw,freq,pos,64,1/64,1,2048,True,source,mapping,request,context,64,flags,lanes)
        def check(got):
            ref=reference_query(q,raw,freq,pos,64,1/64,1,2048)
            for a,b in zip(got[:7],ref):assert torch.equal(a,b)
            blocks=torch.where(source<0,2147483647,source).sort(-1).values
            assert torch.equal(got[7],blocks)
            assert torch.equal(got[8],(mapping[0,::64]//64).int())
            unique=((blocks[:,1:]!=blocks[:,:-1]) | (blocks[:,1:]==2147483647)).all(-1)
            assert bool(((flags!=1)|unique).all())
        for kind in ["ordered","reverse","random","duplicates","negative"]:
            source.copy_(torch.randperm(2048,device="cuda",dtype=torch.int32)[None,:].expand(rows,-1))
            if kind=="ordered":source.copy_(source.sort(-1).values)
            elif kind=="reverse":source.copy_(source.sort(-1,descending=True).values)
            elif kind=="duplicates":source[:,1]=source[:,0]
            elif kind=="negative":source[:,::13]=-1
            check(run())
            records.append(dict(kind="short-combined-"+kind,rows=rows,lanes=lanes,wide=wide,status="bitwise PASS"))
            prepared=m.prepare_prefill_sparse_candidates(source,2048,mapping,request,context,1,64,row_flags=flags)
            ref_blocks=torch.where(source<0,2147483647,source).sort(-1).values
            assert torch.equal(prepared[0],ref_blocks) and torch.equal(prepared[1],(mapping[0,::64]//64).int())
            unique=((ref_blocks[:,1:]!=ref_blocks[:,:-1]) | (ref_blocks[:,1:]==2147483647)).all(-1)
            assert bool(((flags!=1)|unique).all())
            records.append(dict(kind="short-candidates-"+kind,rows=rows,wide=wide,status="bitwise PASS"))
        originals=[v.clone() for v in (q,source,mapping,freq)]
        run();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=run()
        source.copy_(source.flip(1));source[:,1]=source[:,0];mapping.copy_(mapping.flip(1));q.add_(0.5);freq.copy_(freq.conj())
        graph.replay();torch.cuda.synchronize();check(captured)
        for dst,ref in zip((q,source,mapping,freq),originals):dst.copy_(ref)
        graph.replay();torch.cuda.synchronize();check(captured)
        records.append(dict(kind="short-combined-dynamic-graph-A-B-A",rows=rows,status="bitwise PASS"))
        m.prepare_prefill_sparse_candidates(source,2048,mapping,request,context,1,64,row_flags=flags)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):prepared=m.prepare_prefill_sparse_candidates(source,2048,mapping,request,context,1,64,row_flags=flags)
        for mutate in [True,False]:
            if mutate:source.copy_(source.flip(1));source[:,1]=source[:,0];mapping.copy_(mapping.flip(1))
            else:source.copy_(originals[1]);mapping.copy_(originals[2])
            graph.replay();torch.cuda.synchronize()
            ref_blocks=torch.where(source<0,2147483647,source).sort(-1).values
            assert torch.equal(prepared[0],ref_blocks) and torch.equal(prepared[1],(mapping[0,::64]//64).int())
            unique=((ref_blocks[:,1:]!=ref_blocks[:,:-1]) | (ref_blocks[:,1:]==2147483647)).all(-1)
            assert bool(((flags!=1)|unique).all())
        records.append(dict(kind="short-candidates-dynamic-graph-A-B-A",rows=rows,status="bitwise PASS"))

    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module-file", type=pathlib.Path)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output", type=pathlib.Path)
    a = p.parse_args()
    if a.module_file:
        name = "lightop.attention.mqa"
        spec = importlib.util.spec_from_file_location(name, str(a.module_file))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    else:
        from lightop.attention import mqa as m
    records = run_tests(m, a.quick)
    result = {
        "status": "PASS",
        "gfx": torch.cuda.get_device_properties(0).gcnArchName,
        "checks": records,
    }
    if a.output:
        a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": "PASS", "cases": len(records)}), flush=True)


if __name__ == "__main__":
    main()
