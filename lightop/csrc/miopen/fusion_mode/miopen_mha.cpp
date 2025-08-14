#include "miopen/miopen-wrapper.h"
#include "miopen/graph/tensor.hpp"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <optional>
#include <cmath>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <tuple>
#include <c10/core/Scalar.h>

namespace at {
namespace native {

static miopenTensorArgument_t setProblemTensorDescriptor(miopenProblem_t& problem,
                                                         miopenTensorArgumentId_t id,
                                                         TensorDescriptor& input,
                                                         void* tensor_data_ptr)
{
    MIOPEN_CHECK(miopenSetProblemTensorDescriptor(problem, id, input.mut_desc()));
    return miopenTensorArgument_t{id, input.get_storage_addr(), tensor_data_ptr};
}

Tensor miopen_scaled_dot_product_attention(const Tensor& query,
                                           const Tensor& key,
                                           const Tensor& value,
                                           const std::optional<Tensor>& attn_mask_,
                                           const double droprate       = 0.0,
                                           const bool is_causal        = false,
                                           std::optional<double> scale = std::nullopt,
                                           const bool enable_gqa       = false)
{

    if(query.dim() != 4 || key.dim() != 4 || value.dim() != 4)
        MIOPEN_THROW("Only support MHA 2d");

    MhaDescriptor mha_descriptor;
    mha_descriptor.SetParams(1 / std::sqrt(query.sizes()[3]));
    miopenProblem_t problem = nullptr;
    MIOPEN_CHECK(miopenCreateMhaProblem(&problem, &mha_descriptor, miopenProblemDirectionForward));

    bool is_bshd       = false;
    const int n        = query.sizes()[0];
    const int head_q   = query.sizes()[1];
    const int seqlen_q = query.sizes()[2];
    const int seqlen_k = key.sizes()[2];
    const int dim_v    = value.sizes()[3];

    Tensor LSE = at::empty({n, head_q, seqlen_q, 1}, query.options().dtype(at::kFloat));
    Tensor O   = !is_bshd ? at::empty_strided({n, head_q, seqlen_q, dim_v},
                                            {head_q * seqlen_q * dim_v, seqlen_q * dim_v, dim_v, 1},
                                            query.options())
                          : at::empty_strided({n, head_q, seqlen_q, dim_v},
                                            {head_q * seqlen_q * dim_v, dim_v, head_q * dim_v, 1},
                                            query.options());

    TensorDescriptor tquery(query), tkey(key), tvalue(value), tLSE(LSE), tO(O);

    std::vector<miopenTensorArgument_t> args;
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaQ, tquery, query.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaK, tkey, key.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaV, tvalue, value.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaO, tO, O.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaLSE, tLSE, LSE.data_ptr()));

    size_t found = 0;
    std::vector<miopenSolution_t> solutions(1);
    miopenHandle_t handle = getMiopenHandle_new();
    MIOPEN_CHECK(miopenFindSolutions(handle, problem, nullptr, solutions.data(), &found, solutions.size()));
    if(found == 0)
        MIOPEN_THROW("no solution found");

    size_t workspace_size = 0;
    MIOPEN_CHECK(miopenGetSolutionWorkspaceSize(solutions[0], &workspace_size));

    Tensor workspace_tensor;
    if(workspace_size > 0)
    {
        workspace_tensor = at::empty(workspace_size, query.options());
    }

    MIOPEN_CHECK(miopenRunSolution(handle,
                                   solutions[0],
                                   args.size(),
                                   args.data(),
                                   workspace_tensor.defined() ? workspace_tensor.data_ptr() : nullptr,
                                   workspace_size));

    return O;
};

} // namespace native

} // namespace at
