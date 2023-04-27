import torch
import torch.nn as nn
from torchsummary import summary


def build_block(in_channel, out_channel, kernel_size, stride=1, maxpool=False):
    padding = kernel_size//2
    block = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                          nn.BatchNorm2d(out_channel),
                          nn.LeakyReLU(0.1, inplace=True))
    if maxpool:
        block.add_module("maxpool2d", nn.MaxPool2d(kernel_size=2, stride=2))
    return block


class YOLOv1(nn.Module):
    def __init__(self, num_classes=20):
        super(YOLOv1, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(build_block(3, 64, 7, stride=2, maxpool=True))
        self.layer2 = nn.Sequential(build_block(64, 192, 3, maxpool=True))

        self.layer3 = nn.Sequential(build_block(192, 128, 1),
                                    build_block(128, 256, 3),
                                    build_block(256, 256, 1),
                                    build_block(256, 512, 3, maxpool=True))

        self.layer4 = nn.Sequential(build_block(512, 256, 1),
                                    build_block(256, 512, 3),
                                    build_block(512, 256, 1),
                                    build_block(256, 512, 3),
                                    build_block(512, 256, 1),
                                    build_block(256, 512, 3),
                                    build_block(512, 256, 1),
                                    build_block(256, 512, 3),
                                    build_block(512, 512, 1),
                                    build_block(512, 1024, 3, maxpool=True))

        self.layer5 = nn.Sequential(build_block(1024, 512, 1),
                                    build_block(512, 1024, 3),
                                    build_block(1024, 512, 1),
                                    build_block(512, 1024, 3),
                                    build_block(1024, 1024, 3),
                                    build_block(1024, 1024, 3, stride=2))

        self.layer6 = nn.Sequential(build_block(1024, 1024, 3),
                                    build_block(1024, 1024, 3))

        self.layer7 = nn.Sequential(nn.Flatten(),
                                    nn.Linear(7 * 7 * 1024, 4096, bias=True),
                                    nn.LeakyReLU(0.1),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, 7 * 7 * (self.num_classes + 5 * 2), bias=True),
                                    nn.Sigmoid(),
                                    )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.view(-1, 7, 7, self.num_classes + 5 * 2)
        return x


if __name__ == '__main__':
    net = YOLOv1()
    print(summary(net, (3, 448, 448)))

    x = torch.randn((1, 3, 448, 448))
    print(net(x).shape)




