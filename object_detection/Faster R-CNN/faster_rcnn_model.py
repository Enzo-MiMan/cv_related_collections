import torch.nn as nn
import torchvision


class FasterRCNN(nn.Module):
    def __init__(self, backbone, rpn, resize_and_padding):
        super(FasterRCNN, self).__init__()
        self.resize_and_padding = resize_and_padding
        self.backbone = backbone
        self.rpn = rpn
        # self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.resize_and_padding(images, targets)
        features = self.backbone(images.tensors)  # 将图像输入backbone得到特征图
        proposals, proposal_losses = self.rpn(images, features, targets)
        # detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        # detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        #
        # losses = {}
        # losses.update(detector_losses)
        # losses.update(proposal_losses)
        # return self.eager_outputs(losses, detections)
        return proposals, proposal_losses







