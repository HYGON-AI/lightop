"""GPU checks for the C4 logical-union indexing contract."""

import unittest

import torch

from lightop.kvcache.topk_workspace import dsv4_pack_c4_topk_workspace


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class C4TopKWorkspaceTests(unittest.TestCase):
    def test_union_order_duplicates_swa_and_padding_on_current_stream(self):
        stream = torch.cuda.Stream()
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype), torch.cuda.stream(stream):
                # Duplicate physical IDs must not merge distinct logical rows.
                ids = torch.tensor([7, 3, 7, 9, 0], device="cuda", dtype=dtype)
                backing = torch.full((4, 9), 12345, device="cuda", dtype=torch.int32)
                rows = backing[1:3, :6]
                rows.copy_(torch.tensor(
                    [[2, 0, 2, 5, 6, 999], [4, 2, -1, 7, 888, 777]],
                    device="cuda", dtype=torch.int32,
                ))
                lengths = torch.tensor([5, 4], device="cuda", dtype=torch.int32)
                before = backing.clone()
                packed, remapped, count = dsv4_pack_c4_topk_workspace(
                    ids, rows, lengths, 3, capacity=5
                )
            stream.synchronize()
            torch.testing.assert_close(packed.cpu(), torch.tensor([7, 7, 0, -1, -1], dtype=torch.int32))
            torch.testing.assert_close(remapped.cpu(), torch.tensor(
                [[1, 0, 1, 5, 6, -1], [2, 1, -1, 7, -1, -1]], dtype=torch.int32
            ))
            self.assertEqual(count.dtype, torch.int32)
            self.assertEqual(count.shape, (1,))
            self.assertTrue(count.is_cuda)
            torch.testing.assert_close(
                count, torch.tensor([3], device="cuda", dtype=torch.int32)
            )
            torch.testing.assert_close(backing, before)

    def test_empty_union_and_padded_query(self):
        ids = torch.arange(16, device="cuda", dtype=torch.int32)
        rows = torch.tensor([[16, 17, 123], [0, 1, 2]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
        packed, remapped, count = dsv4_pack_c4_topk_workspace(ids, rows, lengths, 2)
        torch.testing.assert_close(packed.cpu(), torch.full((4,), -1, dtype=torch.int32))
        torch.testing.assert_close(remapped.cpu(), torch.tensor([[4, 5, -1], [-1, -1, -1]], dtype=torch.int32))
        torch.testing.assert_close(
            count, torch.tensor([0], device="cuda", dtype=torch.int32)
        )

    def test_runtime_length_changes_and_capacity_validation(self):
        for n in (1, 16, 287, 1151, 3027, 4096):
            ids = torch.arange(n, device="cuda", dtype=torch.int32)
            rows = torch.tensor([[n - 1, n - 1, n]], device="cuda", dtype=torch.int32)
            lengths = torch.tensor([3], device="cuda", dtype=torch.int32)
            packed, remapped, count = dsv4_pack_c4_topk_workspace(ids, rows, lengths, 2)
            capacity = min(n, 2)
            self.assertEqual(packed[0].item(), n - 1)
            torch.testing.assert_close(
                count, torch.tensor([1], device="cuda", dtype=torch.int32)
            )
            torch.testing.assert_close(remapped.cpu(), torch.tensor([[0, 0, capacity]], dtype=torch.int32))
            with self.assertRaisesRegex(RuntimeError, "safe union bound"):
                dsv4_pack_c4_topk_workspace(ids, rows, lengths, 2, capacity=capacity - 1)


if __name__ == "__main__":
    unittest.main()
