# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact CP permutation regression and synchronized target benchmark."""
import argparse
import json
import statistics

import torch
from lightop.attention.cp_output import interleave_cp_output_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(20261109)
    device = torch.device('cuda:0')
    stream = torch.cuda.Stream()
    count = 0
    for rows, cp in ((32368, 8), (32768, 8), (32000, 8), (640, 8), (24, 3)):
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32,
                      torch.int64, torch.float16, torch.bfloat16, torch.float32,
                      torch.float64):
            for width in (1, 3, 128, 512):
                # Compare storage bits, including floating NaNs and signed zero.
                nbytes = rows * width * torch.empty((), dtype=dtype).element_size()
                data = torch.randint(0, 256, (nbytes,), device=device,
                                     dtype=torch.uint8).view(dtype).reshape(rows, width)
                expected = data.view(cp, rows // cp, width).transpose(0, 1).reshape_as(data)
                ptr = data.data_ptr()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    actual = interleave_cp_output_(data, cp)
                stream.synchronize()
                assert actual.data_ptr() == ptr
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                count += 1
    # Storage offsets and odd row widths retain the existing generic fallback.
    data = torch.randint(0, 256, (32368 * 7 + 1,), device=device,
                         dtype=torch.uint8)[1:].reshape(32368, 7)
    expected = data.reshape(8, 4046, 7).transpose(0, 1).reshape_as(data)
    assert torch.equal(interleave_cp_output_(data, 8), expected)
    count += 1
    data = torch.randn(32368, 512, dtype=torch.bfloat16, device=device)
    interleave_cp_output_(data, 8)  # Warm plan before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        interleave_cp_output_(data, 8)
    for _ in range(3):
        original = torch.randn_like(data)
        expected = original.view(8, 4046, 512).transpose(0, 1).reshape_as(data)
        data.copy_(original)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(data.view(torch.uint8), expected.view(torch.uint8))
        count += 1
    result = []
    for rows in (32368, 32768):
        for width in (512, 1024, 1536, 2048, 5120):
            x = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
            for _ in range(10):
                interleave_cp_output_(x, 8)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_bytes = torch.cuda.memory_allocated()
            times = []
            for _ in range(100):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                interleave_cp_output_(x, 8)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) * 1000)
            mean = statistics.mean(times)
            result.append(dict(rows=rows, width=width, dtype=str(x.dtype),
                               mean_us=mean, p50_us=statistics.median(times),
                               p90_us=sorted(times)[89],
                               cv=statistics.pstdev(times) / mean,
                               GB_s=(2 * x.nbytes) / mean / 1000,
                               peak_extra_bytes=torch.cuda.max_memory_allocated()-base_bytes))
    if args.profile:
        x = torch.zeros(32368, 512, device=device, dtype=torch.bfloat16)
        interleave_cp_output_(x, 8)
        torch.cuda.synchronize()
        for _ in range(10):
            interleave_cp_output_(x, 8)
        torch.cuda.synchronize()
    print(json.dumps({'correctness_cases': count, 'benchmark': result}))


if __name__ == '__main__':
    main()
