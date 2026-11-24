// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// HCU MMAC 统一封装（gfx928 / gfx936 / gfx938 / gfx92a / gfx946）
//
// 三个指令族，按 DTK 的 target-feature 划分：
//
//   gfx92a        MMOP5  ：无 lit/lts 参数；INT8 操作数为 2×32bit(<2 x int>)，
//                          必须走 _i8_clamp 变体（int64 版 _i32_16x16x32_* 在 92a 不存在）
//   gfx938/gfx946 MMOP4+ ：需显式 lit/lts；INT8 操作数同为 <2 x int>，走 _lit_clamp_lts
//                          （lit/clamp/lts 全 false，与仓库其它 946 代码保持一致）
//   gfx928/gfx936 MMOP2  ：INT8 为 packed 64bit(int64_t)，走 _i32_16x16x32i8
//
// 一律使用 __builtin_hcu_* 接口。禁止再使用 __builtin_amdgcn_mmac_*：
// 它们在 gfx938 / gfx92a / gfx946 上均不可编译（且 hcu builtin 手册未收录），
// 详见 /opt/dtk/llvm/include/clang/Basic/BuiltinsLingShu.def 的 target-feature 谓词。

#if defined(__gfx92a__)

using hcu_mmac_intx2 = int __attribute__((ext_vector_type(2)));

#define HCU_MMAC_F32_F16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_f16((a), (b), (c))
#define HCU_MMAC_F32_BF16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_bf16((a), (b), (c))
#define HCU_MMAC_F32_TF32(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x8_tf32((a), (b), (c))
// 92a 没有 int64 形参的 INT8 MMAC，只能 bitcast 成 <2 x int> 走 clamp 变体
#define HCU_MMAC_I32_I8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32_i8_clamp( \
        __builtin_bit_cast(hcu_mmac_intx2, (a)), \
        __builtin_bit_cast(hcu_mmac_intx2, (b)), (c), 0)

#elif defined(__gfx938__) || defined(__gfx946__)

using hcu_mmac_intx2 = int __attribute__((ext_vector_type(2)));

#define HCU_MMAC_F32_F16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts((a), (b), (c), false, false)
#define HCU_MMAC_F32_BF16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts((a), (b), (c), false, false)
#define HCU_MMAC_F32_TF32(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts((a), (b), (c), false, false)
#define HCU_MMAC_I32_I8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts( \
        __builtin_bit_cast(hcu_mmac_intx2, (a)), \
        __builtin_bit_cast(hcu_mmac_intx2, (b)), (c), false, false, false)

#else  // gfx928 / gfx936（MMOP2）

#define HCU_MMAC_F32_F16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_f16((a), (b), (c))
#define HCU_MMAC_F32_BF16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_bf16((a), (b), (c))
#define HCU_MMAC_F32_TF32(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x8_tf32((a), (b), (c))
#define HCU_MMAC_I32_I8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32i8((a), (b), (c))

#endif
