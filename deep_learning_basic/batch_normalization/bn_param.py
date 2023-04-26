import torch.nn as nn
from torchsummary import summary

model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1),
                      nn.BatchNorm2d(16),
                      nn.ReLU())

print(summary(model, (3, 224, 224), 8))



