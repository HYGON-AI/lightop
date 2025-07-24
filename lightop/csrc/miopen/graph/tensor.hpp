#include "miopen/miopen-wrapper.h"
#include "miopen/Descriptors.h"
#include <ATen/ATen.h>
#include <vector>

namespace at {
namespace native {

std::pair<BackendDescriptor, size_t>
CreatGraphTensor(const TensorDescriptor& tensor, miopenDataType_t data_type, const bool is_virtual);

std::vector<int64_t> GetConvForwardOutputDim(const TensorDescriptor& input,
                                             const TensorDescriptor& weight,
                                             const std::vector<int64_t>& pads,
                                             const std::vector<int64_t>& strides,
                                             const std::vector<int64_t>& dilations);

class InitTensor
{
private:
    std::vector<miopenTensorDescriptor_t> descVector;
    std::vector<miopenTensorArgument_t> args;
    miopenProblem_t problem = nullptr;
    std::map<miopenTensorArgumentId_t, at::Tensor> tensors;

public:
    void setTensor(miopenTensorArgumentId_t id, at::Tensor& tensor)
    {
        TensorDescriptor input(tensor);
        MIOPEN_CHECK(miopenSetProblemTensorDescriptor(problem, id, input));
        descVector.push_back(&input);
        args.emplace_back();
        args.back().id     = id;
        args.back().buffer = tensor.data_ptr();
        if(tensors.count(id) != 0)
            raise ValueError("The ID has been inserted multiple times");
        tensors[id] = std::move(tensor);
    }

    std::vector<miopenTensorArgument_t> getArgs() { return args; }
    miopenProblem_t getProblem() { return problem; }
    std::vector<miopenTensorDescriptor_t> getDescVector() { return descVector; }
    std::map<miopenTensorArgumentId_t, at::Tensor> getTensors() { return tensors; }
};
} // namespace native
} // namespace at