#include "miopen/miopen-wrapper.h"
#include "miopen/mha/tensor.hpp"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <tuple>

namespace at {
namespace native {

Tensor mha(const Tensor& query,
           const Tensor& key,
           const Tesnor& value,
           const int n              = 1,
           const int head_q         = 4,
           const int head_k         = -1,
           const int seqlen_q       = 128,
           const int seqlen_k       = -1,
           const int dim_q          = 128,
           const int dim_v          = -1,
           const float droprate     = 0.5,
           const std::string layout = "BHSD")
{

    std::size_t found;
    MhaDescriptor mha_descriptor;
    miopenProblem_t problem = nullptr;
    std::vector<miopenSolution_t> solutions(1);
    bool is_bshd = layout.compare("BSHD") == 0;

    mha_descriptor.SetParams(1);
    MIOPEN_CHECK(miopenCreateMhaProblem(&problem, &mha_descriptor, miopenProblemDirectionForward));

    Tensor padding_mask = at::empty({n, 1, 1, 1}, query.options().dtype(at::kInt)).fill_(s_k);
    Tensor bias         = at::empty({n, 1, s, s_k}, query.options().dtype(at::kFloat)).fill_(0);
    Tensor descaleQ     = at::empty({1, 1, 1, 1}, query.options().dtype(at::kFloat)).fill_(1);
    Tensor descaleK     = at::empty_like(descaleQ);
    Tensor descaleV     = at::empty_like(descaleQ);
    Tensor descaleS     = at::empty_like(descaleQ);
    Tensor scaleS       = at::empty_like(descaleQ);
    Tensor scaleO       = at::empty_like(descaleQ);
    Tensor dropProb     = at::empty({1, 1, 1, 1}, query.options().dtype(at::kFloat)).fill_(droprate);
    Tensor dropSeed     = at::empty({1, 1, 1, 1}, query.options().dtype(at::kLong)).fill_(0xAAFFFFFFFFull);
    Tensor dropOffset   = at::empty({1, 1, 1, 1}, query.options().dtype(at::kLong)).fill_(1);
    Tensor M            = at::empty({n, head_q, seqlen_q, 1}, query.options().dtype(at::kFloat));
    Tensor ZInv         = at::empty_like(M);
    Tensor LSE          = at::empty_like(M);
    Tensor AmaxO        = at::empty({1, 1, 1, 1}, query.options().dtype(at::kFloat));
    Tensor AmaxS        = at::empty_like(AmaxO);
    Tensor O            = !is_bshd ? at::empty_strided({n, head_q, seqlen_q, dim_v},
                                            {n * head_q * seqlen_q, seqlen_q * dim_v, dim_v, 1},
                                            query.options())
                                   : at::empty_strided({n, head_q, seqlen_q, dim_v},
                                            {n * head_q * seqlen_q, dim_v, head_q * dim_v, 1},
                                            query.options());

    InitTensor x;
    x.setTensor(miopenTensorMhaQ, query);
    x.setTensor(miopenTensorMhaK, key);
    x.setTensor(miopenTensorMhaV, value);
    x.setTensor(miopenTensorMhaPaddingMask, padding_mask);
    x.setTensor(miopenTensorMhaBias, bias);
    x.setTensor(miopenTensorMhaDescaleQ, descaleQ);
    x.setTensor(miopenTensorMhaDescaleK, descaleK);
    x.setTensor(miopenTensorMhaDescaleV, descaleV);
    x.setTensor(miopenTensorMhaDescaleS, descaleS);
    x.setTensor(miopenTensorMhaScaleS, scaleS);
    x.setTensor(miopenTensorMhaScaleO, scaleO);
    x.setTensor(miopenTensorMhaDropoutProbability, dropProb);
    x.setTensor(miopenTensorMhaDropoutSeed, dropSeed);
    x.setTensor(miopenTensorMhaDropoutOffset, dropOffset);
    x.setTensor(miopenTensorMhaO, O);
    x.setTensor(miopenTensorMhaM, M);
    x.setTensor(miopenTensorMhaZInv, ZInv);
    x.setTensor(miopenTensorMhaLSE, LSE);
    x.setTensor(miopenTensorMhaAmaxO, AmxaO);
    x.setTensor(miopenTensorMhaAmaxS, AmaxS);

    std::vector<miopenTensorArgument_t> args         = x.getArgs();
    std::vector<miopenTensorDescriptor_t> descVector = x.getDescVector();
    for(size_t i = 0; i < descVector.size(); ++i)
    {
        args[i].descriptor = &descVector[i];
    }

    miopenHandle_t handle = getMiopenHandle_new();

    MIOPEN_CHECK(miopenFindSolutions(handle, problem, nullptr, solutions.data(), &found, solutions.size()));

    miopenSolution_t solution = solutions[0];

    size_t workspace_size = 0;
    MIOPEN_CHECK(miopenGetSolutionWorkspaceSize(solution, &workspace_size));

    Tensor workspace_tensor;
    workspace_tensor = at::empty(workspace_size, query.options());

    MIOPEN_CHECK(
        miopenRunSolution(handle, solutions[0], args.size(), args.data(), workspace_tensor.ptr(), workspace_size));

    std::map<miopenTensorArgumentId_t, at::Tensor> tensors = x.getTensors();

    return tensors[miopenTensorMhaO];
};

} // namespace native

} // namespace at
