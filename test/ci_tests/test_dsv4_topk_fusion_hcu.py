# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Independent logical-union checks for the gfx936 fused workspace path."""

import unittest

import torch

from lightop.kvcache.topk_workspace import dsv4_pack_sparse_topk_workspace


def reference(ids, indices, lengths, topk, swa_rows):
    ids, indices, lengths = ids.cpu(), indices.cpu(), lengths.cpu()
    n = ids.numel()
    q, width = indices.shape
    capacity = min(n, q * topk)
    valid = torch.arange(width)[None] < lengths[:, None]
    compressed = valid & (indices >= 0) & (indices < n)
    union = torch.unique(indices[compressed].long(), sorted=True)
    packed = torch.full((capacity,), -1, dtype=torch.int32)
    packed[: union.numel()] = ids[union].int()
    mapped = torch.full_like(indices, -1)
    if union.numel():
        mapped[compressed] = torch.searchsorted(union, indices[compressed].long()).int()
    swa = valid & (indices >= n)
    if swa_rows >= 0:
        swa &= indices - n < swa_rows
    swa &= indices.long() - n + capacity <= torch.iinfo(torch.int32).max
    mapped[swa] = (indices[swa].long() - n + capacity).int()
    return packed, mapped, torch.tensor([union.numel()], dtype=torch.int32)


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class HcuTopKFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith(("gfx936","gfx938")):
            raise unittest.SkipTest("gfx936/gfx938-specific fusion checks")
        torch.set_num_threads(1)

    def inputs(self, n, q, width, topk, dtype, noncontiguous=False, empty=False):
        generator = torch.Generator().manual_seed(1731 + n + q + width)
        ids = (torch.arange(n) % 257).to(dtype)  # Physical aliases stay logically distinct.
        indices = torch.empty((q, width), dtype=torch.int32)
        indices[:, :topk] = torch.randint(-2, n, (q, topk), generator=generator, dtype=torch.int32)
        if width > topk:
            indices[:, topk:] = torch.randint(n, n + 140, (q, width - topk), generator=generator, dtype=torch.int32)
        lengths = torch.randint(0, width + 3, (q,), generator=generator, dtype=torch.int32)
        if empty:
            lengths.zero_()
        ids = ids.cuda()
        if noncontiguous:
            backing = torch.full((q + 1, width + 3), 999999, dtype=torch.int32, device="cuda")
            view = backing[1:, 1:width + 1]
            view.copy_(indices)
            indices = view
        else:
            indices = indices.cuda()
        return ids, indices, lengths.cuda()

    def check(self, inputs, topk, swa_rows=128):
        expected = reference(*inputs, topk, swa_rows)
        before = tuple(t.clone() for t in inputs)
        result = dsv4_pack_sparse_topk_workspace(*inputs, topk, swa_rows=swa_rows)
        torch.cuda.synchronize()
        for actual, wanted in zip(result, expected):
            torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
        for actual, wanted in zip(inputs, before):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        self.assertEqual(result[2].untyped_storage().nbytes(), 4)
        return result

    def test_scan_boundaries_and_scalar_layouts(self):
        for dtype in [torch.int32, torch.int64]:
            for n in [1,31,32,33,255,256,257,1023,1024,1025,32767,65537,131073,1048577]:
                for noncontiguous in [False, True]:
                    with self.subTest(dtype=dtype,n=n,noncontiguous=noncontiguous):
                        self.check(self.inputs(n, 3, 9, 5, dtype, noncontiguous), 5)

    def test_vectorized_widths_queries_and_full_padding(self):
        for q in [1,3,4,5,63,64,65,257]:
            for width, topk in [(4,2),(128,64),(640,512),(768,512)]:
                for empty in [False,True]:
                    with self.subTest(q=q,width=width,empty=empty):
                        self.check(self.inputs(65537,q,width,topk,torch.int32,empty=empty),topk)

    def test_real_chunk_dense_and_overlapping_unions(self):
        for n in [16384,32768,65536,131072]:
            for overlap in [False,True]:
                with self.subTest(n=n,overlap=overlap):
                    ids,indices,lengths=self.inputs(n,4096,640,512,torch.int32)
                    lengths.fill_(640)
                    if overlap:
                        indices[:,:512]=torch.arange(512,dtype=torch.int32,device="cuda")[None]
                    else:
                        indices[:,:512]=(torch.arange(4096,device="cuda")[:,None]*512+torch.arange(512,device="cuda")[None])%n
                    self.check((ids,indices,lengths),512)

    def test_nondefault_streams_and_graph_refresh(self):
        inputs=self.inputs(65537,65,640,512,torch.int64)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            dsv4_pack_sparse_topk_workspace(*inputs,512,swa_rows=128)
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            result=dsv4_pack_sparse_topk_workspace(*inputs,512,swa_rows=128)
        pointers=[t.data_ptr() for t in result]
        original=[t.clone() for t in inputs]
        for changed in [False,True,False]:
            if changed:
                inputs[1][:,:512].remainder_(137)
                inputs[2].zero_()
                inputs[2][::2]=640
            else:
                for dst,src in zip(inputs,original):dst.copy_(src)
            stream.wait_stream(torch.cuda.current_stream())
            graph.replay();stream.synchronize()
            expected=reference(*inputs,512,128)
            for actual,wanted in zip(result,expected):
                torch.testing.assert_close(actual.cpu(),wanted,rtol=0,atol=0)
            self.assertEqual(pointers,[t.data_ptr() for t in result])

    def test_prefix_union_and_single_hole(self):
        for n in [1025, 65537]:
            for span in [1, 512, 1024]:
                for hole in [False, True]:
                    with self.subTest(n=n,span=span,hole=hole):
                        ids,indices,lengths=self.inputs(n,65,640,512,torch.int64)
                        logical=torch.arange(65*512,device="cuda").reshape(65,512)%span
                        if hole:
                            # Cardinality stays unchanged, but a hole at zero
                            # makes logical rank differ from logical row.
                            logical+=1
                        indices[:,:512]=logical.int()
                        lengths.fill_(640)
                        self.check((ids,indices,lengths),512)

    def test_identity_mapping_bounds(self):
        n=1024
        ids,indices,lengths=self.inputs(n,4,640,512,torch.int32)
        indices[:,:512]=torch.arange(4*512,device="cuda").reshape(4,512).remainder(n).int()
        lengths.fill_(640)
        bounds=[n-1,n,n+127,n+128,torch.iinfo(torch.int32).max,-1,torch.iinfo(torch.int32).min]
        indices[:,512:519]=torch.tensor(bounds,dtype=torch.int32,device="cuda")
        for swa_rows in [-1,0,128]:
            with self.subTest(swa_rows=swa_rows):
                self.check((ids,indices,lengths),512,swa_rows=swa_rows)

    def test_bitmap_scan_transitions_and_last_row(self):
        for n in [31,32,33,32768,32769,65536,65537,131072,131073,1048576,1048577]:
            with self.subTest(n=n):
                ids,indices,lengths=self.inputs(n,5,9,5,torch.int64)
                indices[:,:5]=torch.tensor([0,min(31,n-1),min(32,n-1),n//2,n-1],dtype=torch.int32,device="cuda")
                lengths.fill_(9)
                self.check((ids,indices,lengths),5)

    def test_ragged_word_runs_and_repeated_masks(self):
        for q in [1,65,257]:
            with self.subTest(q=q):
                ids,indices,lengths=self.inputs(4096,q,640,512,torch.int32)
                columns=torch.arange(512,device="cuda")[None,:]
                queries=torch.arange(q,device="cuda")[:,None]
                # Seven-element runs cross wave boundaries; repeated word
                # groups stay separated, and only odd bits are selected.
                logical=((columns//7)%8)*32+((queries+columns*7)%16)*2+1
                indices[:,:512]=logical.int()
                lengths.fill_(640)
                self.check((ids,indices,lengths),512)


if __name__ == "__main__":
    unittest.main()
