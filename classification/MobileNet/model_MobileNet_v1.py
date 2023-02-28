import torch.nn as nn
from torchsummary import summary


class MobileNet_v1(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        self.model = nn.Sequential(
            self.conv_bn(3, 32, 2),
            self.depth_sep_conv(32, 64, 1),
            self.depth_sep_conv(64, 128, 2),
            self.depth_sep_conv(128, 128, 1),
            self.depth_sep_conv(128, 256, 2),
            self.depth_sep_conv(256, 256, 1),
            self.depth_sep_conv(256, 512, 2),
            self.depth_sep_conv(512, 512, 1),
            self.depth_sep_conv(512, 512, 1),
            self.depth_sep_conv(512, 512, 1),
            self.depth_sep_conv(512, 512, 1),
            self.depth_sep_conv(512, 512, 1),
            self.depth_sep_conv(512, 1024, 2),
            self.depth_sep_conv(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def conv_bn(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def depth_sep_conv(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    net = MobileNet()
    print(summary(net, (3, 224, 224)))


