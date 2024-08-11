import torch
import my_project_cuda_extension


if __name__ == '__main__':
    x = torch.tensor([[1.0, 2.0], [3.3, 1.1]], device='cuda')
    y = torch.tensor([[3.0, 4.0], [1.2, 4.3]], device='cuda')

    result = my_project_cuda_extension.add(x, y)
    print(result)