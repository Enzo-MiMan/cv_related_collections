import torchvision.models as models
from torchsummary import summary
import torch


# https://pytorch.org/vision/stable/models.html
# alexnet = models.alexnet(weights=None)
# resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# print(resnet50)


# -----------------------------------------------------------
# 保存模型 / 保存模型+参数
# -----------------------------------------------------------

# resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 1、仅保存模型的参数
# torch.save(resnet50.state_dict(), 'resnet50_weight.pth')

# 2、保存模型 + 参数
# torch.save(resnet50, 'resnet50.pth')


# -----------------------------------------------------------
# 加载模型 / 加载模型+参数
# -----------------------------------------------------------

# 1、加载模型+参数
# net = torch.load("resnet50.pth")
# print(net)

# 2、已有模型，加载预训练参数
resnet50 = models.resnet50(weights=None)

resnet50.load_state_dict(torch.load('resnet50_weight.pth'))


