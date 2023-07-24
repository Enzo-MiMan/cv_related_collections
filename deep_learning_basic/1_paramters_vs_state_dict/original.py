import torch
import torch.nn as nn


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1,  bias=False)

        self.weight = torch.ones(10, 10)
        self.bias = torch.zeros(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x * self.weight + self.bias
        return x


net = MyModule()

for name, param in net.named_parameters():
    print(name, param.shape)

print('\n', '*'*40, '\n')

for key, val in net.state_dict().items():
    print(key, val.shape)

