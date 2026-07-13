#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/Dispatch.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Half.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/native/Resize.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/ATenHIPGeneral.h>
#include <ATen/hip/HIPDevice.h>
#include <ATen/hip/HIPContext.h>


namespace at {  namespace native {
Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options);
}
}

#ifdef TORCH_VERSION_OVER_110

#define TORCH_META_FUNC_FINAL(name)                                                                       \
struct structured_##name##_functional final : public at::native::structured_##name {                      \
    void set_output_strided(                                                                              \
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,                                       \
        TensorOptions options, DimnameList names                                                          \
    ) override {                                                                                          \
        auto current_device = guard_.current_device();                                                    \
        if (C10_UNLIKELY(current_device.has_value())) {                                                   \
          TORCH_INTERNAL_ASSERT(*current_device == options.device(),                                      \
            "structured kernels don't support multi-device outputs");                                     \
        } else {                                                                                          \
          guard_.reset_device(options.device());                                                          \
        }                                                                                                 \
        outputs_[output_idx] = create_out(sizes, strides, options);                                       \
        if (!names.empty()) {                                                                             \
          namedinference::propagate_names(*outputs_[output_idx], names);                                  \
        }                                                                                                 \
    }                                                                                                     \
    void set_output_raw_strided(                                                                          \
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,                                       \
        TensorOptions options, DimnameList names                                                          \
    ) override {                                                                                          \
        auto current_device = guard_.current_device();                                                    \
        if (C10_UNLIKELY(current_device.has_value())) {                                                   \
          TORCH_INTERNAL_ASSERT(*current_device == options.device(),                                      \
            "structured kernels don't support multi-device outputs");                                     \
        } else {                                                                                          \
          guard_.reset_device(options.device());                                                          \
        }                                                                                                 \
        outputs_[output_idx] = create_out(sizes, strides, options);                                       \
        if (!names.empty()) {                                                                             \
          namedinference::propagate_names(*outputs_[output_idx], names);                                  \
        }                                                                                                 \
    }                                                                                                     \
    const Tensor& maybe_get_output(int64_t output_idx) override {                                         \
        return *outputs_[output_idx];                                                                     \
    }                                                                                                     \
    std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;                                                \
    c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard_;                                                  \
};

#define TORCH_META_FUNC_FINAL_SET_OUTPUT(name)                                                            \
struct structured_##name##_functional final : public at::native::structured_##name {                      \
    void set_output_strided(                                                                              \
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,                                       \
        TensorOptions options, DimnameList names                                                          \
    ) override {                                                                                          \
        auto current_device = guard_.current_device();                                                    \
        if (C10_UNLIKELY(current_device.has_value())) {                                                   \
          TORCH_INTERNAL_ASSERT(*current_device == options.device(),                                      \
            "structured kernels don't support multi-device outputs");                                     \
        } else {                                                                                          \
          guard_.reset_device(options.device());                                                          \
        }                                                                                                 \
        outputs_[output_idx] = create_out(sizes, strides, options);                                       \
        if (!names.empty()) {                                                                             \
          namedinference::propagate_names(*outputs_[output_idx], names);                                  \
        }                                                                                                 \
        at::native::structured_##name::set_output_raw_strided(output_idx, sizes, strides, options, names);\
    }                                                                                                     \
    void set_output_raw_strided(                                                                          \
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,                                       \
        TensorOptions options, DimnameList names                                                          \
    ) override {                                                                                          \
        auto current_device = guard_.current_device();                                                    \
        if (C10_UNLIKELY(current_device.has_value())) {                                                   \
          TORCH_INTERNAL_ASSERT(*current_device == options.device(),                                      \
            "structured kernels don't support multi-device outputs");                                     \
        } else {                                                                                          \
          guard_.reset_device(options.device());                                                          \
        }                                                                                                 \
        outputs_[output_idx] = create_out(sizes, strides, options);                                       \
        if (!names.empty()) {                                                                             \
          namedinference::propagate_names(*outputs_[output_idx], names);                                  \
        }                                                                                                 \
        at::native::structured_##name::set_output_raw_strided(output_idx, sizes, strides, options, names);\
    }                                                                                                     \
    const Tensor& maybe_get_output(int64_t output_idx) override {                                         \
        return *outputs_[output_idx];                                                                     \
    }                                                                                                     \
    std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;                                                \
    c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard_;                                                  \
};


#else 


#define TORCH_META_FUNC_FINAL(name)                                                                       \
struct structured_##name##_functional final : public at::native::structured_##name {                      \
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,                           \
                    TensorOptions options, DimnameList names) override {                                  \
        auto current_device = guard_.current_device();                                                    \
        if (C10_UNLIKELY(current_device.has_value())) {                                                   \
          TORCH_INTERNAL_ASSERT(*current_device == options.device(),                                      \
            "structured kernels don't support multi-device outputs");                                     \
        } else {                                                                                          \
          guard_.reset_device(options.device());                                                          \
        }                                                                                                 \
        outputs_[output_idx] = create_out(sizes, strides, options);                                       \
        if (!names.empty()) {                                                                             \
          namedinference::propagate_names(*outputs_[output_idx], names);                                  \
        }                                                                                                 \
    }                                                                                                     \
    const Tensor& maybe_get_output(int64_t output_idx) override {                                         \
        return *outputs_[output_idx];                                                                     \
    }                                                                                                     \
    std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;                                                \
    c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard_;                                                  \
};

#define TORCH_META_FUNC_FINAL_SET_OUTPUT(name)                                                            \
struct structured_##name##_functional final : public at::native::structured_##name {                      \
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,                           \
                    TensorOptions options, DimnameList names) override {                                  \
        auto current_device = guard_.current_device();                                                    \
        if (C10_UNLIKELY(current_device.has_value())) {                                                   \
          TORCH_INTERNAL_ASSERT(*current_device == options.device(),                                      \
            "structured kernels don't support multi-device outputs");                                     \
        } else {                                                                                          \
          guard_.reset_device(options.device());                                                          \
        }                                                                                                 \
        outputs_[output_idx] = create_out(sizes, strides, options);                                       \
        if (!names.empty()) {                                                                             \
          namedinference::propagate_names(*outputs_[output_idx], names);                                  \
        }                                                                                                 \
        at::native::structured_##name::set_output(output_idx, sizes, strides, options, names);            \
    }                                                                                                     \
    const Tensor& maybe_get_output(int64_t output_idx) override {                                         \
        return *outputs_[output_idx];                                                                     \
    }                                                                                                     \
    std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;                                                \
    c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard_;                                                  \
};


#endif
