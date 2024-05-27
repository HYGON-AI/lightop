#include <ATen/AccumulateType.h>
template <typename scalar_t>
struct CELU
{
    using T_ACC=at::acc_type<scalar_t, true>;
    __device__ T_ACC forward(scalar_t x) {
        return x < 0 ? alpha * std::expm1(static_cast<T_ACC>(x)/alpha) : static_cast<T_ACC>(x);
    }

    __device__ T_ACC backward(scalar_t dY,scalar_t x) {
        return x < 0 ? std::exp(static_cast<T_ACC>(x)/alpha)*static_cast<T_ACC>(dY) : static_cast<T_ACC>(dY);
    }
    T_ACC alpha;
};

template <typename scalar_t>
struct PassThrough
{
    __device__ scalar_t forward(scalar_t x) {
        return x ;
    }
    __device__ scalar_t backward(scalar_t dY,scalar_t x) {
        return dY;
    }
};