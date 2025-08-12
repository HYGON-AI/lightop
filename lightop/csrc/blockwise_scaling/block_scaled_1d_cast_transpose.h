

namespace at { 
namespace native {

////////////////////////////////////////////////////////////////////////////////////////////////////
  struct uint16 {
  uint4 u;
  uint4 v;
  uint4 s;
  uint4 t;
  };


  struct uint8 {
    uint4 u;
    uint4 v;
  };

  template <int BYTES>
  struct BytesToType {};

  template <>
  struct BytesToType<64> {
    using Type = uint16;
    static_assert(sizeof(Type) == 64);
  };

  template <>
  struct BytesToType<32> {
    using Type = uint8;
    static_assert(sizeof(Type) == 32);
  };

  template <>
  struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
  };

  template <>
  struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
  };

  template <>
  struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
  };

  template <>
  struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
  };

  template <>
  struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
  };

////////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename Elt_type, uint32_t NUM_ELT>
  struct Vec {
    enum { BYTES = NUM_ELT * sizeof(Elt_type) };

    using Vec_type = typename BytesToType<BYTES>::Type;
    using type = Elt_type;

    using Alias_type = union {
      Vec_type vec;
      Elt_type elt[NUM_ELT];
    };

    Alias_type data;
 
    __HOST_DEVICE__ Vec& operator=(const Vec& rhs) {
      data.vec = rhs.data.vec;
      return *this;
    }

    template <typename S>
    inline __device__ void to(Vec<S, NUM_ELT> &other) {  // NOLINT(*)
    #pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        other.data.elt[it] = S(this->data.elt[it]);
      }
    }

    template <typename Op>
    inline __device__ void assign(const Op &op) {
    #pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        this->data.elt[it] = op(it);
      }
    }

    // Pointer is cast to vector type
    inline __device__ void load_from(const void *base_ptr, size_t idx = 0) {
      this->data.vec = static_cast<const Vec_type *>(base_ptr)[idx];
    }

    // Pointer is cast to vector type
    inline __device__ void store_to(void *base_ptr, size_t idx = 0) const {
      static_cast<Vec_type *>(base_ptr)[idx] = this->data.vec;
    }

    // Pointer is cast to element type. Loads min(count, NUM_ELT)
    // elements and any remaining elements are set to zero.
    inline __device__ void load_from_elts(const void *base_ptr, size_t idx = 0,
                                          size_t count = NUM_ELT) {
      const Elt_type *elt_ptr = static_cast<const Elt_type *>(base_ptr) + idx;
      if (count < NUM_ELT || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0) {
  #pragma unroll
        for (int it = 0; it < NUM_ELT; it++) {
          this->data.elt[it] = (it < count ? elt_ptr[it] : Elt_type(0.f));
        }
      } else {
        this->load_from(elt_ptr);
      }
    }

    // Pointer is cast to element type. Stores min(count, NUM_ELT)
    // elements.
    inline __device__ void store_to_elts(void *base_ptr, size_t idx = 0,
                                        size_t count = NUM_ELT) const {
      Elt_type *elt_ptr = static_cast<Elt_type *>(base_ptr) + idx;
      if (count < NUM_ELT || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0) {
      #pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        if (it < count) {
          elt_ptr[it] = this->data.elt[it];
        }
      }
      } else {
        this->store_to(elt_ptr);
      }
        }

    inline __device__ void clear() {
    #pragma unroll
        for (int it = 0; it < NUM_ELT; it++) {
          this->data.elt[it] = Elt_type(0.f);
        }
      }
    };

////////////////////////////////////////////////////////////////////////////////////////////////////

  __device__ __forceinline__ float compute_scale_from_amax(float amax, float max_fp8,
                                                          bool force_pow_2_scales, float epsilon,
                                                          float value_for_inf) {
    // NOTE: NAN amax evaluates false for <, handled further down.
    if (amax < epsilon) {
      amax = epsilon;
    }

    float scale = 1.f;

    if (isinf(amax) || amax == 0.f || isnan(amax)) {
      return scale;
    }

    // Here we don't use "scale = max_fp8 / amax" because it has different results with/without
    // "--use_fast_math".
    // "__fdiv_rn" has the same behavior with "max_fp8 / amax" when not using fast math.
    scale = __fdiv_rn(max_fp8, amax);

    // The amax is too small that the scale becoming infinite in FP32. In other word,
    // the scale is not representable in FP32.
    if (isinf(scale)) {
      // use fp32 max to represent the scale
      scale = value_for_inf;
    }
    if (force_pow_2_scales) {
      uint32_t scale_bits = *reinterpret_cast<uint32_t *>(&scale);
      scale_bits &= 0xFF800000;
      // If the exponent was zero, we have a logic error.
      __builtin_assume(scale_bits != 0 || scale == 0.0);
      __builtin_assume(scale_bits != 0x80000000);
      scale = *reinterpret_cast<float *>(&scale_bits);
    }

    return scale;
  }

  // Calculate the quantization scale for an individual data element
  // given the amax(abs(tile)) value for a given quantization tile.
  //
  //
  // Arguments:
  // IType: data type of the tensor being quantized (float or bf16)
  // OType: quantized data type (e4m3 or e5m2)
  // amax: The evaluation of amax(abs(tile)) for the quantization tile.
  // eps: An epsilon used as a floor for amax.
  // pow_2_scaling: Whether to force the scale to be a power of 2.
  template <typename IType, typename OType>
  __device__ __forceinline__ float compute_scale_from_types(const float amax, const float eps,
                                                            const float pow_2_scaling) {
    constexpr float fp8_max = TypeInfo<OType>::max_finite_value;
    // NOTE: We're relying on compute_scale_from_amax to have behavior where it
    // clips the mantissa of the max_finite_value if power of 2 scaling applies.
    constexpr float value_for_inf = TypeInfo<IType>::max_finite_value;
    return compute_scale_from_amax(amax, fp8_max, pow_2_scaling, eps, value_for_inf);
  }


  }  // 

}  // 