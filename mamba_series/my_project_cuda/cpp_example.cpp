#include <torch/extension.h>
#include "utils.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

torch::Tensor add(const torch::Tensor& x, const torch::Tensor& y) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");

    CHECK_SHAPE(x, 2, 2);
    CHECK_SHAPE(y, 2, 2);

    return add_fwd(x, y);
}

 PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
     m.def("add", &add);
 }