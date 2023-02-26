import torch
import math
import torch.nn as nn


class Resize_and_Padding(nn.Module):
    def __init__(self, min_size, max_size):
        super(Resize_and_Padding, self).__init__()
        self.min_size = min_size  # 指定图像的最小边长范围
        self.max_size = max_size

    def forward(self, images, targets=None):
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None
            image, target_index = self.resize_image_and_bbox(image, target_index)   # resize image and boxes
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self.padding_images(images)  # 将batch 中的图像padding 到相同的尺寸

        image_sizes_list = []
        for image_size in image_sizes:
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def resize_image_and_bbox(self, image, target):
        im_shape = torch.tensor(image.shape[-2:])
        # ======================= resize image =======================
        image_min_size = float(torch.min(im_shape))
        image_max_size = float(torch.max(im_shape))
        scale_factor = self.min_size / image_min_size  # 按照短边进行缩放，计算缩放比例

        # 根据短边的缩放比例来缩放长边，长边缩放后的结果是否超出预设的 max_size
        if image_max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / image_max_size  # 若超出预设 max_size, 则按照长边的比例进行缩放

        image = torch.nn.functional.interpolate(image[None], scale_factor=scale_factor, mode="bilinear",
                                                recompute_scale_factor=True, align_corners=False)[0]
        # ======================= resize bbox =======================
        boxes = target["boxes"]
        ratios_height = torch.as_tensor(image.shape[1] / im_shape[0], dtype=torch.float32, device=boxes.device)
        ratios_width = torch.as_tensor(image.shape[2] / im_shape[1], dtype=torch.float32, device=boxes.device)
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        xmin = xmin * ratios_width
        xmax = xmax * ratios_width
        ymin = ymin * ratios_height
        ymax = ymax * ratios_height
        target["boxes"] = torch.stack((xmin, ymin, xmax, ymax), dim=1)
        return image, target

    def padding_images(self, images, size_divisible=32):
        image_size = [img.shape for img in images]
        max_size = torch.max(torch.tensor(image_size), dim=0)[0]
        stride = float(size_divisible)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + [v.item() for v in max_size]
        '''以images[0]为基，创建带 padding 的 新tensor，
        是为了新创建的 tensor 在相同的 device 上'''
        batched_imgs = images[0].new_full(batch_shape, 0)

        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        return batched_imgs


class ImageList(object):
    def __init__(self, tensors, image_sizes):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)













