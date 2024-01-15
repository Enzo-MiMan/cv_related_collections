import numpy as np
import os
import torch
import torch.nn as nn
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wandb.init(project='LeNet5', resume=True)

train_batch_size = 16
test_batch_size = 32
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# 下载数据 & 导入数据
train_set = mnist.MNIST("./", train=True, download=True, transform=transform)
test_set = mnist.MNIST("./", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

class LeNet5(nn.Module):
    """ 使用sequential构建网络，Sequential()函数的功能是将网络的层组合到一起 """

    def __init__(self, in_channel, output):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=5, stride=1, padding=2),  # (6, 28, 28)
                                    nn.Tanh(),
                                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0))  # (6, 14, 14))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # (16, 10, 10)
                                    nn.Tanh(),
                                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0))  # (16, 5, 5)
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
wandb.watch(model, log='all', log_graph=True)

lr = 0.01
num_epoches = 5
momentum = 0.8

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

artifact = wandb.use_artifact("lenet5-mnist:v0")
model_dir = artifact.download()
model_path = os.path.join(model_dir, "lenet5.pth")
model.load_state_dict(torch.load(model_path))

for epoch in range(num_epoches, 10):

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
            wandb.log({'loss': loss.item(), 'acc_rate': acc_rate})


# torch.save(model.state_dict(), "lenet5_v2.pth")
# artifact = wandb.Artifact('lenet5-mnist', type='model_param')
# artifact.add_file('./lenet5_v2.pth')
# wandb.log_artifact(artifact)
wandb.finish()


