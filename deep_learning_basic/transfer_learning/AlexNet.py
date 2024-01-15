import torch.nn as nn
import torchvision.models as models


alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
# print(alexnet)

# 1、----- 删除网络的最后一层 -----
# del alexnet.classifier
# del alexnet.classifier[6]
# print(alexnet)


# 2、----- 删除网络的最后多层 -----
# alexnet.classifier = alexnet.classifier[:-2]
# print(alexnet)


# 3、----- 修改网络的某一层 -----
alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=1024)
# print(alexnet)


# 4、----- 网络添加层, 每次添加一层 -----
# alexnet.classifier.add_module('7', nn.ReLU(inplace=True))
# alexnet.classifier.add_module('8', nn.Linear(in_features=1024, out_features=20))
# print(alexnet)


# 4、----- 网络添加层，一次添加多层 -----
block = nn.Sequential(nn.ReLU(inplace=True),
                      nn.Linear(in_features=1024, out_features=20))
alexnet.add_module('block', block)
print(alexnet)

