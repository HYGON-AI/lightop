#include "miopen/graph/tensor.hpp"
#include <cstddef>
#include <miopen/Exceptions.h>
#include <vector>
#include <stdint.h>

namespace at {
namespace native {

static size_t GetGraphTensorUUID()
{
    static size_t init_uuid = 1337;
    return init_uuid++;
}
std::pair<BackendDescriptor, size_t> CreatGraphTensor(const TensorDescriptor& tensor,
                                                      miopenDataType_t data_type,
                                                      miopenTensorLayout_t tensor_layout,
                                                      const bool is_virtual)
{
    static const int64_t alignment = 4;

    int dims = tensor.size();

    std::vector<int> lens(dims);
    std::vector<int> strides(dims);
    tensor.get(data_type, lens.data(), strides.data());

    std::vector<int64_t> input_lens(lens.cbegin(), lens.cend());
    std::vector<int64_t> input_strides(strides.cbegin(), strides.cend());

    BackendDescriptor tensor_desc(MIOPEN_BACKEND_TENSOR_DESCRIPTOR);
    tensor_desc.set(MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &data_type);
    tensor_desc.set(MIOPEN_ATTR_TENSOR_DATA_LAYOUT, MIOPEN_TYPE_DATA_LAYOUT, 1, &tensor_layout);
    tensor_desc.set(MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT64, input_lens.size(), input_lens.data());
    tensor_desc.set(MIOPEN_ATTR_TENSOR_STRIDES, MIOPEN_TYPE_INT64, input_strides.size(), input_strides.data());
    auto uuid = GetGraphTensorUUID();
    tensor_desc.set(MIOPEN_ATTR_TENSOR_UNIQUE_ID, MIOPEN_TYPE_INT64, 1, &uuid);
    tensor_desc.set(MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT, MIOPEN_TYPE_INT64, 1, &alignment);
    tensor_desc.set(MIOPEN_ATTR_TENSOR_IS_VIRTUAL, MIOPEN_TYPE_BOOLEAN, 1, &is_virtual);
    tensor_desc.finalize();
    return std::make_pair(std::move(tensor_desc), uuid);
}
std::vector<int64_t> GetConvForwardOutputDim(const TensorDescriptor& input,
                                             const TensorDescriptor& weight,
                                             const std::vector<int64_t>& pads,
                                             const std::vector<int64_t>& strides,
                                             const std::vector<int64_t>& dilations)
{
    if(pads.size() != 2)
    {
        MIOPEN_THROW("Only support conv 2d");
    }
    std::vector<int> output_dims(4);
    std::unique_ptr<miopenConvolutionDescriptor_t, void (*)(miopenConvolutionDescriptor_t*)> convolution_desc(
        new miopenConvolutionDescriptor_t, [](miopenConvolutionDescriptor_t* ptr) {
            if(ptr)
                miopenDestroyConvolutionDescriptor(*ptr);
        });

    MIOPEN_CHECK(miopenCreateConvolutionDescriptor(convolution_desc.get()));
    std::vector<int> temp_pad(pads.cbegin(), pads.cend());
    std::vector<int> temp_stride(strides.cbegin(), strides.cend());
    std::vector<int> temp_dilation(dilations.cbegin(), dilations.cend());

    MIOPEN_CHECK(miopenInitConvolutionNdDescriptor(*convolution_desc,
                                                   temp_pad.size(),
                                                   temp_pad.data(),
                                                   temp_stride.data(),
                                                   temp_dilation.data(),
                                                   miopenConvolution));

    MIOPEN_CHECK(miopenGetConvolutionForwardOutputDim(*convolution_desc,
                                                      input.desc(),
                                                      weight.desc(),
                                                      &output_dims[0],
                                                      &output_dims[1],
                                                      &output_dims[2],
                                                      &output_dims[3]));
    return std::vector<int64_t>(output_dims.cbegin(), output_dims.cend());
}
} // namespace native
} // namespace at