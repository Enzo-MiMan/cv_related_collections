import torch
import torch.nn as nn
import  torchsummary


class Yolo1_Model(nn.Module):
    def __init__(self):
        super(Yolo1_Model, self).__init__()

        self.features = nn.Sequential(
            Conv_BN(3, 64, 7, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv_BN(64, 192, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv_BN(192, 128, 1, 1),
            Conv_BN(128, 256, 3, 1),
            Conv_BN(256, 256, 1, 1),
            Conv_BN(256, 512, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv_BN(512, 256, 1, 1),
            Conv_BN(256, 512, 3, 1),
            Conv_BN(512, 256, 1, 1),
            Conv_BN(256, 512, 3, 1),
            Conv_BN(512, 256, 1, 1),
            Conv_BN(256, 512, 3, 1),
            Conv_BN(512, 256, 1, 1),
            Conv_BN(256, 512, 3, 1),
            Conv_BN(512, 512, 1, 1),
            Conv_BN(512, 1024, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv_BN(1024, 512, 1, 1),
            Conv_BN(512, 1024, 3, 1),
            Conv_BN(1024, 512, 1, 1),
            Conv_BN(512, 1024, 3, 1),
            Conv_BN(1024, 1024, 3, 1),
            Conv_BN(1024, 1024, 3, 2),

            Conv_BN(1024, 1024, 3, 1),
            Conv_BN(1024, 1024, 3, 1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(1024, 4096)
        self.linear2 = nn.Linear(4096, 1470)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(-1, 30, 7, 7)
        return x


class Conv_BN(nn.Module):
    def __init__(self, in_chan, out_chan, k_size, stride):
        super(Conv_BN, self).__init__()
        padding = (k_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=k_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# net = Yolo1_Model()
# print(net)
# print(torchsummary.summary(net, (3, 448, 448)))









