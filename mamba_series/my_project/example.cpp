#include <torch/extension.h>

torch::Tensor add(const torch::Tensor& x, const torch::Tensor& y) {
  return x + y;
}

 PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
     m.def("add", &add, "A function that adds two numbers");
 }