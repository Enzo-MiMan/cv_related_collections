#include <torch/extension.h>

torch::Tensor add_fwd(const torch::Tensor& x, const torch::Tensor& y);