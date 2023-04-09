import torch
import torch.nn as nn
from torchsummary import summary


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, dilation_rate=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=dilation_rate, bias=False, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, bottleneck_num, num_classes=21, replace_conv=None):
        super(ResNet, self).__init__()
        if replace_conv is None:
            replace_conv = [False, False, False]
        self.in_channel = 64
        self.dilation_rate = 1

        if len(replace_conv) != 3:
            raise ValueError("replace_stride_with_dilation should be None " "or a 3-element tuple, got {}".format(replace_conv))

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 64, 128, 256, 512 分别是 每一层第一个block里第一个 convolution 的 out_channel
        self.layer1 = self._make_block(64, bottleneck_num[0])
        self.layer2 = self._make_block(128, bottleneck_num[1], stride=2, replace_conv=replace_conv[0])
        self.layer3 = self._make_block(256, bottleneck_num[2], stride=2, replace_conv=replace_conv[1])
        self.layer4 = self._make_block(512, bottleneck_num[3], stride=2, replace_conv=replace_conv[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_block(self, out_channel, block_num, stride=1,  replace_conv=False):
        downsample = None
        previous_dilation_rate = self.dilation_rate
        if replace_conv:
            self.dilation_rate *= stride
            stride = 1

        # 每个 layer 的第一个 block, downsample 表示跨层连接时是否需要下采样
        if stride != 1 or self.in_channel != out_channel * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=out_channel*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel*4)
            )

        layers = []
        layers.append(Bottleneck(self.in_channel, out_channel, stride, downsample, previous_dilation_rate))
        self.in_channel = out_channel * 4

        # 每个 layer 的第二个 block 到最后一个 block
        for _ in range(1, block_num):
            layers.append(Bottleneck(self.in_channel, out_channel, dilation_rate=self.dilation_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet50(**kwargs):
    model = ResNet([3, 4, 6, 3], **kwargs)
    return model


if __name__ == '__main__':
    print(resnet50(replace_conv=[False, True, True]))
    # model = resnet50(replace_stride_with_dilation=[False, True, True])
    # print(summary(model, (3, 224, 224)))