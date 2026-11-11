// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Match DTK du_mma.hpp: gfx92a uses MMOP5 rather than the gfx928/936
// MMOP2 builtins. The lane layout is unchanged; INT8 operands change from
// packed 64-bit scalars to two 32-bit registers, with saturation disabled.
#if defined(__gfx92a__)
using hcu_mmac_intx2 = int __attribute__((ext_vector_type(2)));
#define HCU_MMAC_F32_F16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_f16((a), (b), (c))
#define HCU_MMAC_F32_BF16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_bf16((a), (b), (c))
#define HCU_MMAC_F32_TF32(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x8_tf32((a), (b), (c))
#define HCU_MMAC_I32_I8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32_i8_clamp( \
        __builtin_bit_cast(hcu_mmac_intx2, (a)), \
        __builtin_bit_cast(hcu_mmac_intx2, (b)), (c), 0)
#else
#define HCU_MMAC_F32_F16(a, b, c) \
    __builtin_amdgcn_mmac_f32_16x16x16f16((a), (b), (c))
#define HCU_MMAC_F32_BF16(a, b, c) \
    __builtin_amdgcn_mmac_f32_16x16x16bf16((a), (b), (c))
#define HCU_MMAC_F32_TF32(a, b, c) \
    __builtin_amdgcn_mmac_f32_16x16x8tf32((a), (b), (c))
#define HCU_MMAC_I32_I8(a, b, c) \
    __builtin_amdgcn_mmac_i32_16x16x32i8((a), (b), (c))
#endif
