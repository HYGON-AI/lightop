# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Selected-row dequantization against the existing floating-point kernel."""
import unittest

import torch

from lightop.kvcache.topk_workspace import (
    _get_workspace_native,
    dsv4_dequantize_sparse_topk_out,
)


def make_cache(page_size, exhaustive=False, padding=0):
    physical = 256 if exhaustive else ((37 + page_size - 1) // page_size) * page_size
    pages = physical // page_size
    stride = page_size * 584 + padding
    cache = torch.zeros((pages, 1, stride), dtype=torch.uint8, device="cuda")
    data = cache[:, 0, :page_size * 576].view(pages, page_size, 576)
    scale = cache[:, 0, page_size * 576:page_size * 584].view(pages, page_size, 8)
    raw = torch.arange(448, device="cuda", dtype=torch.int32)[None].expand(physical, -1)
    raw = (raw & 255) if exhaustive else raw % 127
    data[:, :, :448] = raw.byte().reshape(pages, page_size, 448)
    values = torch.arange(physical, device="cuda", dtype=torch.int32) if exhaustive else torch.full((physical,), 128, device="cuda", dtype=torch.int32)
    scale[:, :, :7] = values.byte().view(pages, page_size, 1)
    if exhaustive:
        rope = ((torch.arange(physical * 64, device="cuda", dtype=torch.int32) * 7919) & 65535).short().view(torch.bfloat16)
    else:
        rope = (torch.arange(physical * 64, device="cuda", dtype=torch.float32).remainder(127) / 31).bfloat16()
    data[:, :, 448:] = rope.view(torch.uint8).reshape(pages, page_size, 128)
    return cache, physical


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class SelectedDequantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith(("gfx936","gfx938")):
            raise unittest.SkipTest("gfx936/gfx938 only")
        cls.native = _get_workspace_native()

    def assert_bits(self, actual, expected):
        a, e = actual.contiguous().view(torch.int16).cpu(), expected.contiguous().view(torch.int16).cpu()
        if not torch.equal(a, e):
            bad = torch.nonzero(a != e)
            examples = [(index.tolist(), int(a[tuple(index)]), int(e[tuple(index)])) for index in bad[:16]]
            self.fail("BF16 bit mismatches: %d; examples=%s" % (bad.shape[0], examples))

    def reference(self, cache, ids, page_size, fnuz):
        out = torch.empty((ids.numel(), 1, 512), dtype=torch.bfloat16, device="cuda")
        count = torch.tensor([ids.numel()], dtype=torch.int32, device="cuda")
        self.native.dequantize_k_cache_paged_hcu_counted(cache, ids, out, page_size, fnuz, count)
        return out

    def test_exhaustive_fp8_scale_bits(self):
        cache, physical = make_cache(64, exhaustive=True)
        ids = torch.arange(physical, dtype=torch.int32, device="cuda")
        selected = ids.flip(0).contiguous()
        count = torch.tensor([physical], dtype=torch.int32, device="cuda")
        for fnuz in [False, True]:
            with self.subTest(fnuz=fnuz):
                expected = self.reference(cache, ids, 64, fnuz)
                actual = torch.empty_like(expected)
                self.native.dequantize_k_cache_paged_hcu_selected(cache, ids, selected, count, actual, 64, fnuz)
                torch.cuda.synchronize()
                self.assert_bits(actual, expected)

    def test_all_fp8_byte_pairs(self):
        # Packed arithmetic must not carry from the low BF16 into the high
        # BF16. Exercise every pair, including mixed zeros, signs and NaNs.
        n, page_size = 65536, 64
        pages = n // page_size
        cache = torch.zeros((pages, 1, page_size * 584), dtype=torch.uint8, device="cuda")
        data = cache[:, 0, :page_size * 576].view(pages, page_size, 576)
        scales = cache[:, 0, page_size * 576:].view(pages, page_size, 8)
        pairs = torch.arange(n, dtype=torch.int32, device="cuda").view(pages, page_size, 1)
        data[:, :, :448:2] = (pairs & 255).byte()
        data[:, :, 1:448:2] = (pairs >> 8).byte()
        ids = torch.arange(n, dtype=torch.int32, device="cuda")
        count = torch.tensor([n], dtype=torch.int32, device="cuda")
        actual = torch.empty((n, 1, 512), dtype=torch.bfloat16, device="cuda")
        for scale in [0, 11, 127, 128, 246, 255]:
            scales.fill_(scale)
            for fnuz in [False, True]:
                with self.subTest(scale=scale, fnuz=fnuz):
                    expected = self.reference(cache, ids, page_size, fnuz)
                    self.native.dequantize_k_cache_paged_hcu_selected(cache, ids, ids, count, actual, page_size, fnuz)
                    torch.cuda.synchronize()
                    self.assert_bits(actual, expected)
                    del expected

    def inputs(self, n, page_size, dtype, padding=0):
        cache, physical = make_cache(page_size, padding=padding)
        ids = torch.arange(n, device="cuda").remainder(min(physical, 37)).to(dtype)
        generator = torch.Generator(device="cuda").manual_seed(8401 + n)
        indices = torch.randint(-1, n, (65, 48), generator=generator, dtype=torch.int32, device="cuda")
        indices[:, 32:] = n + torch.arange(16, device="cuda", dtype=torch.int32)
        lengths = torch.randint(0, 49, (65,), generator=generator, dtype=torch.int32, device="cuda")
        return cache, ids, indices, lengths

    def check_selected(self, ids, indices, lengths, output, count, expected):
        i, l = indices.cpu(), lengths.cpu()
        valid = (torch.arange(i.shape[1])[None] < l[:, None]) & (i >= 0) & (i < ids.numel())
        union = torch.unique(i[valid].long()).cuda()
        self.assertEqual(int(count.cpu()[0]), union.numel())
        self.assertEqual(count.untyped_storage().nbytes(), 4)
        selected = torch.zeros(ids.numel(), dtype=torch.bool, device="cuda")
        selected[union] = True
        self.assert_bits(output[selected], expected[selected])
        self.assert_bits(output[~selected], torch.full_like(output[~selected], -3.5))
        return valid.cuda()

    def test_logical_aliases_strides_and_attention_reads(self):
        for page_size in [1, 3, 8, 64]:
            for dtype in [torch.int32, torch.int64]:
                for fnuz in [False, True]:
                    with self.subTest(page_size=page_size,dtype=dtype,fnuz=fnuz):
                        cache,ids,indices,lengths = self.inputs(1025,page_size,dtype,padding=4)
                        before = indices.clone()
                        expected = self.reference(cache,ids,page_size,fnuz)
                        backing = torch.full((1025,1,513),17.0,dtype=torch.bfloat16,device="cuda")
                        output = backing[:,:,:512];output.fill_(-3.5)
                        count = dsv4_dequantize_sparse_topk_out(
                            cache, ids, indices, lengths, output, 32, page_size,
                            fp8_fnuz=fnuz,
                        )
                        torch.cuda.synchronize()
                        valid = self.check_selected(ids,indices,lengths,output,count,expected)
                        self.assert_bits(backing[:,:,512:],torch.full_like(backing[:,:,512:],17.0))
                        torch.testing.assert_close(indices,before,rtol=0,atol=0)
                        packed,remapped,u = self.native.dsv4_pack_sparse_topk_workspace(ids,indices,lengths,32,ids.numel(),16)
                        compact = torch.empty_like(expected)
                        self.native.dequantize_k_cache_paged_hcu_counted(cache,packed,compact,page_size,fnuz,u)
                        # Compare the actual compressed KV values seen by every
                        # valid Attention reference, preserving duplicates.
                        self.assert_bits(output[indices[valid].long()],compact[remapped[valid].long()])

    def test_nondefault_stream_graph_refresh_and_empty_union(self):
        cache,ids,indices,lengths = self.inputs(257,8,torch.int64)
        original = [indices.clone(),lengths.clone(),ids.clone()]
        expected = self.reference(cache,ids,8,False)
        output = torch.full_like(expected,-3.5)
        stream = torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            count = dsv4_dequantize_sparse_topk_out(
                cache, ids, indices, lengths, output, 32, 8, fp8_fnuz=False,
            )
        pointers = (output.data_ptr(),count.data_ptr())
        for state in ['A','empty','B','A']:
            output.fill_(-3.5)
            if state=='empty':lengths.zero_()
            else:
                indices.copy_(original[0]);lengths.copy_(original[1]);ids.copy_(original[2])
                if state=='B':
                    indices[:,:32].remainder_(17)
                    ids.copy_((original[2]+1).remainder(37))
            expected = self.reference(cache,ids,8,False)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):graph.replay()
            stream.synchronize()
            self.check_selected(ids,indices,lengths,output,count,expected)
            self.assertEqual(pointers,(output.data_ptr(),count.data_ptr()))

    def test_rejects_compact_capacity(self):
        cache,ids,indices,lengths = self.inputs(1025,8,torch.int32)
        out = torch.empty((1025,1,512),dtype=torch.bfloat16,device="cuda")
        with self.assertRaisesRegex(RuntimeError,"full-capacity"):
            self.native.dsv4_dequantize_sparse_topk_out(cache,ids,indices[:1],lengths[:1],out,32,8,False)

    def test_local_union_dispatch_boundaries_and_partial_query_groups(self):
        # Cross both capacity cutoffs and the query-count threshold. A final
        # partial CTA must preserve the same logical union and untouched rows.
        for q, n in [(1023, 65535), (1024, 65536), (1025, 65537),
                     (1031, 114688), (1033, 114689), (4097, 33)]:
            with self.subTest(q=q, n=n):
                cache, physical = make_cache(64, padding=4)
                ids = torch.arange(n, device="cuda", dtype=torch.int64).remainder(37)
                generator = torch.Generator(device="cuda").manual_seed(4013 + n + q)
                backing = torch.empty((q, 643), device="cuda", dtype=torch.int32)
                indices = backing[:, :640]
                indices.copy_(torch.randint(-2, n, (q, 640), generator=generator,
                                            device="cuda", dtype=torch.int32))
                indices[:, 512:] = n + torch.arange(128, device="cuda", dtype=torch.int32)
                lengths = torch.randint(0, 641, (q,), generator=generator,
                                        device="cuda", dtype=torch.int32)
                for fnuz in [False, True]:
                    expected = self.reference(cache, ids, 64, fnuz)
                    output = torch.full_like(expected, -3.5)
                    count = dsv4_dequantize_sparse_topk_out(
                        cache, ids, indices, lengths, output, 512, 64, fp8_fnuz=fnuz,
                    )
                    torch.cuda.synchronize()
                    self.check_selected(ids, indices, lengths, output, count, expected)


if __name__ == "__main__":
    unittest.main()
