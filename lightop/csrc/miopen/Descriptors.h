#pragma once

#include "Exceptions.h"

#include "miopen-wrapper.h"
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <memory>
namespace at {
namespace native {

inline miopenDataType_t getDataType(const at::Tensor& t)
{
    auto scalar_type = t.scalar_type();
    if(scalar_type == at::kFloat)
    {
        return miopenFloat;
    }
    else if(scalar_type == at::kHalf)
    {
        return miopenHalf;
    }
    else if(scalar_type == at::kBFloat16)
    {
        return miopenBFloat16;
    }
    else if(scalar_type == at::kByte)
    {
        return miopenInt8;
    }
    else if(scalar_type == at::kInt)
    {
        return miopenInt32;
    }
    else if(scalar_type == at::kLong)
    {
        return miopenInt64;
    }
    else
    {
        throw std::runtime_error("TensorDescriptor only supports float, int32, int8, half and bfloat16 tensors");
    }
}

inline int dataSize(miopenDataType_t dataType)
{
    switch(dataType)
    {
    case miopenHalf: return 2;
    case miopenFloat: return 4;
    case miopenBFloat16: return 2;
    default: return 8;
    }
}

template <typename T, miopenStatus_t (*dtor)(T*)>
struct DescriptorDeleter
{
    void operator()(T* x)
    {
        if(x != nullptr)
        {
            MIOPEN_CHECK(dtor(x));
        }
    }
};

// A generic class for wrapping MIOpen descriptor types.  All you need
// is to give the underlying type the Descriptor_t points to (usually,
// if it's miopenTensorDescriptor_t it points to miopenTensorStruct),
// the constructor and the destructor.  Subclasses are responsible
// for defining a set() function to actually set the descriptor.
//
// Descriptors default construct to a nullptr, and have a descriptor
// initialized the first time you call set() or any other initializing
// function.
template <typename T, miopenStatus_t (*ctor)(T**), miopenStatus_t (*dtor)(T*)>
class Descriptor
{
public:
    // Use desc() to access the underlying descriptor pointer in
    // a read-only fashion.  Most client code should use this.
    // If the descriptor was never initialized, this will return
    // nullptr.
    T* desc() const { return desc_.get(); }
    T* desc() { return desc_.get(); }

    // Use mut_desc() to access the underlying descriptor pointer
    // if you intend to modify what it points to (e.g., using
    // miopenSetFooDescriptor).  This will ensure that the descriptor
    // is initialized.  Code in this file will use this function.
    T* mut_desc()
    {
        init();
        return desc_.get();
    }

    miopenTensorDescriptor_t* get_storage_addr() { return &raw_desc; }

protected:
    void init()
    {
        if(desc_ == nullptr)
        {
            MIOPEN_CHECK(ctor(&raw_desc));
            desc_.reset(raw_desc);
        }
    }

private:
    T* raw_desc;
    mutable std::unique_ptr<T, DescriptorDeleter<T, dtor>> desc_;
};

class TensorDescriptor
    : public Descriptor<miopenTensorDescriptor, &miopenCreateTensorDescriptor, &miopenDestroyTensorDescriptor>
{
public:
    TensorDescriptor() {}
    explicit TensorDescriptor(const at::Tensor& t, size_t pad = 0) { set(t, pad); }

    void set(const at::Tensor& t, size_t pad = 0);
    void set(miopenDataType_t dataType, IntArrayRef sizes, IntArrayRef strides, size_t pad = 0);

    void print();
    void set(miopenDataType_t dataType, int dim, int* size, int* stride)
    {
        MIOPEN_CHECK(miopenSetTensorDescriptor(mut_desc(), dataType, dim, size, stride));
    }
    void set(miopenDataType_t dataType, miopenTensorLayout_t tensor_layout, int* lens, int num_lens)
    {
        MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(mut_desc(), dataType, tensor_layout, lens, num_lens));
    }
    int size() const
    {
        int size = 0;
        MIOPEN_CHECK(miopenGetTensorDescriptorSize(desc(), &size));

        return size;
    }
    void get(miopenDataType_t dataType, int* dimsA, int* stridesA) const
    {
        MIOPEN_CHECK(miopenGetTensorDescriptor(desc(), &dataType, dimsA, stridesA));
    }

    size_t number_size() const
    {
        size_t number_size = 0;
        MIOPEN_CHECK(miopenGetTensorNumBytes(desc(), &number_size));
        return number_size;
    }
};

std::ostream& operator<<(std::ostream& out, const TensorDescriptor& d);

class FilterDescriptor
    : public Descriptor<miopenTensorDescriptor, &miopenCreateTensorDescriptor, &miopenDestroyTensorDescriptor>
{
public:
    void set(const at::Tensor& t, int64_t pad = 0) { set(t, at::MemoryFormat::Contiguous, pad); }

    void set(const at::Tensor& t, const at::MemoryFormat memory_format, int64_t pad = 0);

private:
    void set(miopenDataType_t dataType, int dim, int* size, int* stride)
    {
        MIOPEN_CHECK(miopenSetTensorDescriptor(mut_desc(), dataType, dim, size, stride));
    }
};

struct ConvolutionDescriptor : public Descriptor<miopenConvolutionDescriptor,
                                                 &miopenCreateConvolutionDescriptor,
                                                 &miopenDestroyConvolutionDescriptor>
{
    void set(miopenDataType_t dataType,
             miopenConvolutionMode_t c_mode,
             int dim,
             int* pad,
             int* stride,
             int* upscale /* aka dilation */,
             int groups)
    {
        MIOPEN_CHECK(miopenInitConvolutionNdDescriptor(mut_desc(), dim, pad, stride, upscale, c_mode));
        MIOPEN_CHECK(miopenSetConvolutionGroupCount(mut_desc(), groups));
    }
};

struct RNNDescriptor : public Descriptor<miopenRNNDescriptor, &miopenCreateRNNDescriptor, &miopenDestroyRNNDescriptor>
{
    void set(int64_t hidden_size,
             int64_t num_layers,
             miopenRNNInputMode_t input_mode,
             miopenRNNDirectionMode_t direction,
             miopenRNNMode_t rnn_mode,
             miopenRNNBiasMode_t bias_mode,
             miopenRNNAlgo_t algorithm,
             miopenDataType_t datatype)
    {
        MIOPEN_CHECK(miopenSetRNNDescriptor(
            mut_desc(), hidden_size, num_layers, input_mode, direction, rnn_mode, bias_mode, algorithm, datatype));
    }
};

struct BackendDescriptor
{
public:
    explicit BackendDescriptor(miopenBackendDescriptorType_t type)
    {
        MIOPEN_CHECK(miopenBackendCreateDescriptor(type, &storage_));
        desc_.reset(storage_);
    }
    miopenBackendDescriptor_t desc() const { return desc_.get(); }
    miopenBackendDescriptor_t desc() { return desc_.get(); }

    void set(miopenBackendAttributeName_t attributeName,
             miopenBackendAttributeType_t attributeType,
             int64_t elementCount,
             const void* arrayOfElements)
    {
        MIOPEN_CHECK(miopenBackendSetAttribute(desc(), attributeName, attributeType, elementCount, arrayOfElements));
    }

    void get(miopenBackendAttributeName_t attributeName,
             miopenBackendAttributeType_t attributeType,
             int64_t requestedElementCount,
             int64_t* elementCount,
             void* arrayOfElements) const
    {
        MIOPEN_CHECK(miopenBackendGetAttribute(
            desc(), attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements));
    }

    const miopenBackendDescriptor_t* storage_addr() const { return &storage_; }

    void finalize() { MIOPEN_CHECK(miopenBackendFinalize(desc())); }

private:
    miopenBackendDescriptor_t storage_;
    std::unique_ptr<miopenBackendDescriptor,
                    DescriptorDeleter<miopenBackendDescriptor, &miopenBackendDestroyDescriptor>>
        desc_;
};

struct MhaDescriptor : public miopenMhaDescriptor
{
public:
    void SetParams(float scale_,
                   bool is_inference_     = false,
                   bool use_causal_mask_  = false,
                   bool use_alibi_mask_   = false,
                   bool use_padding_mask_ = false,
                   bool use_attn_bias_    = false)
    {
        scale            = scale_;
        is_inference     = is_inference_;
        use_causal_mask  = use_causal_mask_;
        use_alibi_mask   = use_alibi_mask_;
        use_padding_mask = use_padding_mask_;
        use_attn_bias    = use_attn_bias_;
    }

    void GetParams(float* scale_,
                   bool* is_inference_,
                   bool* use_causal_mask_,
                   bool* use_alibi_mask_,
                   bool* use_padding_mask_,
                   bool* use_attn_bias_) const
    {
        *scale_            = scale;
        *is_inference_     = is_inference;
        *use_causal_mask_  = use_causal_mask;
        *use_alibi_mask_   = use_alibi_mask;
        *use_padding_mask_ = use_padding_mask;
        *use_attn_bias_    = use_attn_bias;
    }

    float GetScale() const { return scale; }

private:
    float scale;
    bool is_inference;
    bool use_causal_mask;
    bool use_alibi_mask;
    bool use_padding_mask;
    bool use_attn_bias;
};

union Constant
{
    float f;
    double d;
    Constant(miopenDataType_t dataType, double value)
    {
        if(dataType == miopenHalf || dataType == miopenFloat || dataType == miopenBFloat16)
        {
            f = static_cast<float>(value);
        }
        else
        {
            d = value;
        }
    }
};

} // namespace native
} // namespace at
