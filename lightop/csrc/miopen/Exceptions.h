#pragma once

#include "miopen-wrapper.h"
#include <string>
#include <stdexcept>
#include <sstream>
#include <exception>
#include <string>

struct THCState;

namespace at {
namespace native {
struct Exception : std::exception
{
    std::string message;
    miopenStatus_t status;
    Exception(const std::string& msg = "") : message(msg), status(miopenStatusUnknownError) {}

    Exception(miopenStatus_t s, const std::string& msg = "") : message(msg), status(s) {}

    Exception SetContext(const std::string& file, int line)
    {

        message = file + ":" + std::to_string(line) + ": " + message;
        return *this;
    }

    // Workaround for HIP, this must be inline
    const char* what() const noexcept override { return message.c_str(); }
};

#define MIOPEN_THROW(...)                                            \
    do                                                               \
    {                                                                \
        throw Exception(__VA_ARGS__).SetContext(__FILE__, __LINE__); \
    } while(false)

inline void MIOPEN_CHECK(miopenStatus_t status)
{
    if(status != miopenStatusSuccess)
    {
        if(status == miopenStatusNotImplemented)
        {
            MIOPEN_THROW(status,
                         std::string(miopenGetErrorString(status)) +
                             ". This error may appear if you passed in a non-contiguous input.");
        }
        MIOPEN_THROW(status, miopenGetErrorString(status));
    }
}

inline void HIP_CHECK(hipError_t error)
{
    if(error != hipSuccess)
    {
        std::string msg("HIP error: ");
        msg += hipGetErrorString(error);
        throw std::runtime_error(msg);
    }
}

} // namespace native
} // namespace at
