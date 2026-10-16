// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// LightOp uses the wavefront width in host-side checks, device code, kernel
// launch dimensions, and template arguments.  Keep this as a project-owned
// compile-time value instead of depending on a private c10 header symbol.
// setup.py may override it with the device's actual warpSize.
#ifndef LIGHTOP_WARP_SIZE
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) || \
    defined(USE_ROCM)
#define LIGHTOP_WARP_SIZE 64
#else
#define LIGHTOP_WARP_SIZE 32
#endif
#endif

// Existing kernels use C10_WARP_SIZE.  Override the Torch-provided spelling
// locally so both old and new Torch headers use the same LightOp value.
#ifdef C10_WARP_SIZE
#undef C10_WARP_SIZE
#endif
#define C10_WARP_SIZE LIGHTOP_WARP_SIZE
