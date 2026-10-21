// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

// Standalone host-side regression; compile with hipcc and run without a GPU.
#include "../../csrc/mqa_logits/beta/bf16/dispatch/auto.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
    using namespace lightop::mqa_bf16;
    MqaBf16Params p{};
    p.m = 32;
    p.n = 64;
    p.h = 8;
    p.d = 128;
    LaunchGrid grid;
    for (int bn : {16, 32, 64}) {
        const KernelConfig paired{KernelKind::SmallKV, 2, bn, 1, 1, true};
        assert(make_grid(p, paired, INT32_MAX, 65535, grid) == (bn == 16));
        if (bn == 16)
            assert(grid.x == 16 && grid.y == 4);
        const KernelConfig single{KernelKind::SmallKV, 1, bn, 1, 1, false};
        assert(make_grid(p, single, INT32_MAX, 65535, grid));
        assert(grid.x == 32 && grid.y == 64 / bn);
    }
    const auto config = resolve(p);
    assert(config.kind == KernelKind::SmallKV && config.paired_queries &&
           config.block_m == 2 && config.block_n == 16);
    assert(make_grid(p, config, INT32_MAX, 65535, grid));
    // Public fast-ReLU regressions must exercise BM32/BN256/GH1.
    for (int dim : {1024, 1025}) {
        p.m = p.n = dim;
        p.h = dim == 1024 ? 64 : 16;
        const auto fast_relu = resolve(p);
        assert(fast_relu.kind == KernelKind::Tiled && fast_relu.block_m == 32 &&
               fast_relu.block_n == 256 && fast_relu.group_heads == 1);
        assert(make_grid(p, fast_relu, INT32_MAX, 65535, grid));
    }
    std::cout << "BF16 config regression passed\n";
}
