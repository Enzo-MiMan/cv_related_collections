import torch
import my_project_cpp_extension

result = my_project_cpp_extension.add(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
print(result)