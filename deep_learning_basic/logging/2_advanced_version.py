import time
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import logging


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_batch_size = 16
test_batch_size = 32
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# 下载数据 & 导入数据
train_set = mnist.MNIST("./", train=True, download=True, transform=transform)
test_set = mnist.MNIST("./", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

# # 抽样查看图片
# examples = enumerate(train_loader)
# batch_index, (example_data, example_label) = next(examples)
# print(type(example_data))   # <class 'torch.Tensor'>
# print(example_data.shape)   # torch.Size([64, 1, 28, 28])

# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray')
#     plt.title("Ground Truth: {}".format(example_label[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


def create_logger():
    # ============================ 1、实例化 logger ============================
    # 实例化一个记录器，并将记录器的名字设为 'training_log'，并将日志级别为 info
    logger = logging.getLogger(name='training_log')
    logger.setLevel(logging.INFO)

    # ============================ 2、定义Handler ============================
    # 创建一个往 console打印输出的 Handler，日志级别为 debug
    consoleHandler = logging.StreamHandler()

    # 再创建一个往文件中打印输出的handler
    fileHandler = logging.FileHandler(filename='mnist.log', mode='w')

    # ============================ 3、定义打印格式 ============================

    simple_formatter = logging.Formatter('%(message)s')

    # ============================ 4、绑定 ============================
    # 让 consoleHandler 使用 简单版日志打印格式
    consoleHandler.setFormatter(simple_formatter)
    # 让 fileHandler 使用 简单版日志打印格式
    fileHandler.setFormatter(simple_formatter)

    # 给记录器绑定上 consoleHandler 和 fileHandler
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger


class LeNet5(nn.Module):
    """ 使用sequential构建网络，Sequential()函数的功能是将网络的层组合到一起 """
    def __init__(self, in_channel, output):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=5, stride=1, padding=2),   # (6, 28, 28)
                                    nn.Tanh(),
                                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0))   # (6, 14, 14))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # (16, 10, 10)
                                    nn.Tanh(),
                                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0))   # (16, 5, 5)
        self.layer3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)  # (120, 1, 1)
        self.layer4 = nn.Sequential(nn.Linear(in_features=120, out_features=84),
                                    nn.Tanh(),
                                    nn.Linear(in_features=84, out_features=output))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(input=x, start_dim=1)
        x = self.layer4(x)
        return x


model = LeNet5(1, 10)
model.to(device)

logger = create_logger()
print(logger)

lr = 0.01
num_epoches = 20
momentum = 0.8

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


for epoch in range(num_epoches):
    logger.info("epoch ：{}".format(epoch))


    if epoch % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.1

    model.train()
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        predict = model(imgs)
        loss = criterion(predict, labels)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accurate rate
        result = torch.argmax(predict, dim=1)
        acc_num = (result == labels).sum().item()
        acc_rate = acc_num / imgs.shape[0]

        if i % 200 == 0:
            logger.info('loss : {}'.format(loss.item()))
            logger.info('acc_rate : {:.3f}'.format(acc_rate))


