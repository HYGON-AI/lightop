// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <string>
#include <sstream>

namespace lightop
{
namespace deep_gemm
{
class LIGHTOPException final : public std::exception {
    std::string message = {};

public:
    explicit LIGHTOPException(const char* name, const char* file, const int line, const std::string& error) {
        message = std::string(name) + " error (" + file + ":" + std::to_string(line) + "): " + error;
    }

    const char* what() const noexcept override {
        return message.c_str();
    }
};

#ifndef LIGHTOP_STATIC_ASSERT
#define LIGHTOP_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef LIGHTOP_HOST_ASSERT
#define LIGHTOP_HOST_ASSERT(cond) \
do { \
if (not (cond)) { \
throw LIGHTOPException("Assertion", __FILE__, __LINE__, #cond); \
} \
} while (0)
#endif

#ifndef LIGHTOP_HOST_UNREACHABLE
#define LIGHTOP_HOST_UNREACHABLE(reason) (throw LIGHTOPException("Assertion", __FILE__, __LINE__, reason))
#endif

#ifndef LIGHTOP_CUDA_DRIVER_CHECK
#define LIGHTOP_CUDA_DRIVER_CHECK(cmd) \
do { \
const auto& e = (cmd); \
if (e != CUDA_SUCCESS) { \
std::stringstream ss; \
const char *name, *info; \
cuGetErrorName(e, &name), cuGetErrorString(e, &info); \
ss << static_cast<int>(e) << " (" << name << ", " << info << ")"; \
throw LIGHTOPException("CUDA driver", __FILE__, __LINE__, ss.str()); \
} \
} while (0)
#endif

#ifndef LIGHTOP_CUDA_RUNTIME_CHECK
#define LIGHTOP_CUDA_RUNTIME_CHECK(cmd) \
do { \
const auto& e = (cmd); \
if (e != cudaSuccess) { \
std::stringstream ss; \
ss << static_cast<int>(e) << " (" << cudaGetErrorName(e) << ", " << cudaGetErrorString(e) << ")"; \
throw LIGHTOPException("CUDA runtime", __FILE__, __LINE__, ss.str()); \
} \
} while (0)
#endif
}
} // namespace deep_gemm
