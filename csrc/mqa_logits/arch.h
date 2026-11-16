// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace lightop::mqa_beta {
// gfx938 instruction and synchronization primitives shared by MQA kernels.
using I2 = int __attribute__((ext_vector_type(2)));
using I4 = int __attribute__((ext_vector_type(4)));
using F2 = float __attribute__((ext_vector_type(2)));
using F4 = float __attribute__((ext_vector_type(4)));
using LdsInt = int __attribute__((address_space(3)));

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
// One MLS instruction and its matching DS format. A kernel owns the LDS layout,
// including how several MLS requests are combined and when their data is ready.
enum class MLSKind { B8_64x32_T1_F1, B8_128x16_T1_F0, B8_64x16_T1_F0 };

struct MLSB8TransDS {
    static constexpr int kDSRow = 3;
    static constexpr int kDSCol = 1;
    static constexpr int kDSAlt = 0;
    static constexpr int kNonmajorFilterShift = 8;
};

template <MLSKind Kind> struct MLSTraits;

template <> struct MLSTraits<MLSKind::B8_64x32_T1_F1> : MLSB8TransDS {
    static constexpr int kRows = 32;
    static constexpr int kLdsBytesPerIssue = 2048;
    static constexpr int kControlWord = 1 << 17;
    static constexpr bool kSupportsTail = true;
};

template <> struct MLSTraits<MLSKind::B8_128x16_T1_F0> : MLSB8TransDS {
    static constexpr int kRows = 16;
    static constexpr int kLdsBytesPerIssue = 2048;
    static constexpr int kControlWord = 0;
    static constexpr bool kSupportsTail = true;
};

template <> struct MLSTraits<MLSKind::B8_64x16_T1_F0> : MLSB8TransDS {
    static constexpr int kRows = 16;
    static constexpr int kLdsBytesPerIssue = 1024;
    static constexpr int kControlWord = 0;
    // The current two-request K path is used only for complete N16 tiles.
    static constexpr bool kSupportsTail = false;
};

template <MLSKind Kind> struct MLSAccessor {
    using Traits = MLSTraits<Kind>;
    using Descriptor = I4;

    // valid_nonmajor is in [1, kRows]. A zero-row tile is handled by the
    // kernel: the filter does not promise to initialize its LDS destination.
    template <bool Tail = false>
    __device__ __forceinline__ static Descriptor make_descriptor(const uint8_t *source,
                                                                  int stride_bytes,
                                                                  int valid_nonmajor = Traits::kRows) {
        static_assert(!Tail || Traits::kSupportsTail, "MLS tail mode is not verified");
        const uintptr_t address = reinterpret_cast<uintptr_t>(source);
        int control = Traits::kControlWord;
        if constexpr (Tail)
            control |= (Traits::kRows - valid_nonmajor) << Traits::kNonmajorFilterShift;
        return {int(address), int(address >> 32), stride_bytes, control};
    }

    __device__ __forceinline__ static void matrix_load(Descriptor descriptor, uint32_t destination) {
        const uint32_t offset = destination | 0x80000000u;
        if constexpr (Kind == MLSKind::B8_64x32_T1_F1)
            asm volatile("s_nop 0\n\tmatrix_load_64x32_b8 %0, %1, moffset:0 t lds"
                         : : "s"(descriptor), "s"(offset) : "memory");
        else if constexpr (Kind == MLSKind::B8_128x16_T1_F0)
            asm volatile("s_nop 0\n\tmatrix_load_128x16_b8 %0, %1, moffset:0 t lds"
                         : : "s"(descriptor), "s"(offset) : "memory");
        else
            asm volatile("s_nop 0\n\tmatrix_load_64x16_b8 %0, %1, moffset:0 t lds"
                         : : "s"(descriptor), "s"(offset) : "memory");
    }

    // OffsetBytes is an immediate relative to the caller's LDS base. It may
    // select data written by a different MLS request in the same stage.
    template <int OffsetBytes = 0>
    __device__ __forceinline__ static I4 ds_read_format(uint32_t source) {
        auto *p = reinterpret_cast<LdsInt *>(uintptr_t(source));
        return __builtin_bit_cast(
            I4, __builtin_hcu_ds_read_matrix_trans_format_i8(
                    p, OffsetBytes, Traits::kDSRow, Traits::kDSCol, Traits::kDSAlt));
    }
};

enum class BufferEmitMode { Builtin, InlineAsm };

template <int BytesPerAccess> struct BufferValue;
template <> struct BufferValue<4> { using type = float; };
template <> struct BufferValue<8> { using type = F2; };
template <> struct BufferValue<16> { using type = F4; };

template <int BytesPerAccess, BufferEmitMode Mode> struct BufferLoadAccessor {
    static_assert(BytesPerAccess == 4 || BytesPerAccess == 8 || BytesPerAccess == 16);
    static_assert(Mode == BufferEmitMode::Builtin || BytesPerAccess == 4,
                  "Only scalar inline-ASM buffer loads are verified");
    using Value = typename BufferValue<BytesPerAccess>::type;
    using Descriptor = I4;
    static constexpr uint32_t kOOBOffset = UINT32_MAX;

    // Raw gfx938 resource descriptor. The UINT32_MAX offset is the tested
    // invalid-request sentinel; every other issued access must be fully valid.
    __device__ __forceinline__ static Descriptor make_descriptor(const void *source) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(source);
        return {int(address), int(address >> 32), -1, 0x00020000};
    }

    // Builtins expose their result dependency to the compiler.
    __device__ __forceinline__ static Value load(Descriptor descriptor, uint32_t offset) {
        static_assert(Mode == BufferEmitMode::Builtin, "Use the voffset/soffset ASM overload");
        if constexpr (BytesPerAccess == 16)
            return __builtin_bit_cast(
                F4, __builtin_amdgcn_buffer_load_dwordx4(descriptor, 0, offset, false, false));
        else if constexpr (BytesPerAccess == 8)
            return __builtin_bit_cast(
                F2, __builtin_amdgcn_buffer_load_dwordx2(descriptor, 0, offset, false, false));
        else
            return __builtin_bit_cast(
                float, __builtin_amdgcn_buffer_load_dword(descriptor, 0, offset, false, false));
    }

    // Scalar soffset keeps the lane's voffset unchanged across prefetched heads.
    // The caller must explicitly wait with the returned VGPR as an operand.
    __device__ __forceinline__ static Value load(Descriptor descriptor, uint32_t voffset,
                                                 uint32_t soffset) {
        static_assert(Mode == BufferEmitMode::InlineAsm && BytesPerAccess == 4,
                      "This ASM load is verified only for one dword");
        float value;
        asm volatile("buffer_load_dword %0, %1, %2, %3 offen"
                     : "=&v"(value)
                     : "v"(voffset), "s"(descriptor), "s"(soffset)
                     : "memory");
        return value;
    }
};

enum class MMADataType { FP8E4M3FN };
enum class MMAEmitMode { InlineAsm, Builtin };

// A and B are already selected pairs of adjacent VGPRs. Only the FP8/LIT
// layout below has been validated with the current C consumer and compiler.
template <MMADataType DataType, bool Lit, bool Lts, MMAEmitMode Mode>
__device__ __forceinline__ F4 mmac(I2 a, I2 b, F4 c) {
    static_assert(DataType == MMADataType::FP8E4M3FN && Lit && !Lts &&
                      Mode == MMAEmitMode::InlineAsm,
                  "Unverified MMAC type, layout or emission mode");
    F4 result;
    asm volatile("v_mmac_f32_16x16x32_fp8_fp8 %0, %1, %2, %3 lit"
                 : "=&v"(result)
                 : "v"(a), "v"(b), "v"(c));
    return result;
}

__device__ __forceinline__ float shuffle_weight(float x, int row) {
    return __builtin_bit_cast(float,
                              __builtin_amdgcn_ds_bpermute(row * 4, __builtin_bit_cast(int, x)));
}

// A single-output ASM avoids this DTK's incorrect extraction of multiple
// FP32 ASM outputs. Native max semantics intentionally replace fmaxf's
// extra input canonicalization; non-finite values follow the ISA operation.
__device__ __forceinline__ float relu(float value) {
    float result;
    asm volatile("v_max_f32 %0, 0, %1" : "=&v"(result) : "v"(value));
    return result;
}

// Counts apply to this wave's outstanding operations, not to a named load.
// Before s_barrier, the caller must satisfy the target's pending-memory and
// LDS ownership requirements. The scheduler barriers only constrain ordering.
template <int VMCnt = -1, int LGKMCnt = -1, bool SBarrier = true>
__device__ __forceinline__ void barrier() {
    static_assert(VMCnt >= -1 && LGKMCnt >= -1);
    __builtin_hcu_sched_barrier(0);
    if constexpr (VMCnt >= 0)
        asm volatile("; MQA_VM_DRAIN\n\ts_waitcnt vmcnt(%0)" : : "i"(VMCnt) : "memory");
    if constexpr (LGKMCnt >= 0)
        asm volatile("s_waitcnt lgkmcnt(%0)" : : "i"(LGKMCnt) : "memory");
    if constexpr (SBarrier)
        asm volatile("; MQA_WG_BARRIER\n\ts_barrier" ::: "memory");
    __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void wait_vm() { barrier<0, -1, false>(); }

// Register operands keep compiler-managed DS/VMEM results live until the
// matching dependency point. A memory clobber alone does not express this.
__device__ __forceinline__ void finish_kv_half(I4 x0, I4 x1, I4 x2, I4 x3) {
    __builtin_hcu_sched_barrier(0);
    asm volatile("; MQA_KV_HALF_READY %0 %1 %2 %3" : : "v"(x0), "v"(x1), "v"(x2), "v"(x3)
                 : "memory");
    __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void finish_weight(float &x) {
    __builtin_hcu_sched_barrier(0);
    asm volatile("; MQA_WEIGHT_READY\n\ts_waitcnt vmcnt(0)" : "+v"(x) : : "memory");
    __builtin_hcu_sched_barrier(0);
}

__device__ __forceinline__ void finish_scales(F4 (&x)[4]) {
    __builtin_hcu_sched_barrier(0);
    asm volatile("; MQA_KV_SCALES_READY\n\ts_waitcnt vmcnt(0)"
                 : "+v"(x[0]), "+v"(x[1]), "+v"(x[2]), "+v"(x[3]) : : "memory");
    __builtin_hcu_sched_barrier(0);
}
#endif
} // namespace lightop::mqa_beta
