// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

// Host-only, exact-value switches for the FP8 kernel selectors. The callable
// returns Kernel. Each runtime expression is evaluated once; invalid sizes
// return nullptr so launch() can reject them.
#define MQA_BOOL_SWITCH(COND, CONST_NAME, ...)                 \
    ([&]() -> Kernel {                                         \
        const bool mqa_switch_value = (COND);                  \
        if (mqa_switch_value) {                                \
            constexpr bool CONST_NAME = true;                 \
            return (__VA_ARGS__)();                            \
        }                                                      \
        constexpr bool CONST_NAME = false;                    \
        return (__VA_ARGS__)();                                \
    }())

#define MQA_SIZE_SWITCH_3(VALUE, CONST_NAME, S0, S1, S2, ...) \
    ([&]() -> Kernel {                                         \
        const auto mqa_switch_value = (VALUE);                 \
        switch (mqa_switch_value) {                            \
        case S0: { constexpr int CONST_NAME = S0; return (__VA_ARGS__)(); } \
        case S1: { constexpr int CONST_NAME = S1; return (__VA_ARGS__)(); } \
        case S2: { constexpr int CONST_NAME = S2; return (__VA_ARGS__)(); } \
        default: return nullptr;                               \
        }                                                      \
    }())

#define MQA_SIZE_SWITCH_4(VALUE, CONST_NAME, S0, S1, S2, S3, ...) \
    ([&]() -> Kernel {                                         \
        const auto mqa_switch_value = (VALUE);                 \
        switch (mqa_switch_value) {                            \
        case S0: { constexpr int CONST_NAME = S0; return (__VA_ARGS__)(); } \
        case S1: { constexpr int CONST_NAME = S1; return (__VA_ARGS__)(); } \
        case S2: { constexpr int CONST_NAME = S2; return (__VA_ARGS__)(); } \
        case S3: { constexpr int CONST_NAME = S3; return (__VA_ARGS__)(); } \
        default: return nullptr;                               \
        }                                                      \
    }())
