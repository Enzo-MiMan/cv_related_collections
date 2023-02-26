import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


def resnet18_backbone():
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # print(backbone)
    backbone = create_feature_extractor(backbone, return_nodes={"layer4": "0"})
    # out = backbone(torch.rand(1, 3, 224, 224))
    # print(out["0"].shape)
    backbone.out_channels = 512
    return backbone


def vgg16_backbone():
    backbone = models.vgg16_bn(pretrained=True)
    # print(backbone)
    backbone = create_feature_extractor(backbone, return_nodes={"features.42": "0"})
    # out = backbone(torch.rand(1, 3, 224, 224))
    # print(out["0"].shape)
    backbone.out_channels = 512
    return backbone


def efficientNetB0_backbone():
    backbone = models.efficientnet_b0(pretrained=True)
    # print(backbone)
    backbone = create_feature_extractor(backbone, return_nodes={"features.5": "0"})
    # out = backbone(torch.rand(1, 3, 224, 224))
    # print(out["0"].shape)
    backbone.out_channels = 112
    return backbone


