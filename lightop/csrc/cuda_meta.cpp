#include "cuda_meta.h"

namespace at{namespace native{
  Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {

  globalContext().lazyInitCUDA();
  if (strides.empty()) {
      return at::native::empty_cuda(sizes, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), options.memory_format_opt());
  } else {
      // TODO: assert options.memory_format_opt() is nullopt (debug only?)
      return at::native::empty_strided_cuda(sizes, strides, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }

}
}
}