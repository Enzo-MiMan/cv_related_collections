import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxes, labels):
        for trans in self.transforms:
            image, bboxes, labels = trans(image, bboxes, labels)
        return image, bboxes, labels


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes,该方法应放在ToTensor后"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, bboxes, labels):
        if random.random() < self.prob:
            image = F.hflip(image)
            width, height = image.size
            bboxes[:, [0, 2]] = width - bboxes[:, [2, 0]]

            # 可视化结果 进行检查
            # image = np.array(image)
            # plt.imshow(image)
            # for i in range(bbox.shape[0]):
            #     xmin, ymin, xmax, ymax = bbox[i, :]
            #     x = [xmin, xmax, xmax, xmin, xmin]
            #     y = [ymin, ymin, ymax, ymax, ymin]
            #     plt.plot(x, y, color='blue', linewidth=2)
            # plt.show()

            return image, bboxes, labels
        return image, bboxes, labels


class RandomScale(object):
    def __init__(self, min_scale, max_scale, prob=0.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob = prob

    def __call__(self, image, bboxes, labels):
        if random.random() < self.prob:
            scale = random.uniform(self.min_scale, self.max_scale)
            width, height = image.size
            image = F.resize(image, [height, int(width * scale)])
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(bboxes)
            boxes = bboxes * scale_tensor
            return image, boxes, labels
        return image, bboxes, labels


class ColorJitter(object):
    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05):
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, bbox, labels):
        image = self.color_jitter(image)
        return image, bbox, labels


class RandomShift(object):
    def __init__(self, padding_value):
        self.padding_value = [int(i*255) for i in padding_value]

    def __call__(self, image, bboxes, labels):
        center = (bboxes[:, 2:] + bboxes[:, :2]) / 2
        image = np.array(image)
        height, width, c = image.shape
        after_shift_image = np.zeros((height, width, c), dtype=image.dtype)
        after_shift_image[:, :, :] = self.padding_value
        shift_x = random.uniform(-width * 0.2, width * 0.2)
        shift_y = random.uniform(-height * 0.2, height * 0.2)

        if shift_x >= 0 and shift_y >= 0:
            after_shift_image[int(shift_y):, int(shift_x):, :] = image[:height - int(shift_y), :width - int(shift_x), :]
        elif shift_x >= 0 and shift_y < 0:
            after_shift_image[:height + int(shift_y), int(shift_x):, :] = image[-int(shift_y):, :width - int(shift_x), :]
        elif shift_x < 0 and shift_y >= 0:
            after_shift_image[int(shift_y):, :width + int(shift_x), :] = image[:height - int(shift_y), -int(shift_x):, :]
        elif shift_x < 0 and shift_y < 0:
            after_shift_image[:height + int(shift_y), :width + int(shift_x), :] = image[-int(shift_y):, -int(shift_x):, :]

        shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
        center = center + shift_xy
        mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
        mask = (mask1 & mask2).view(-1, 1)
        boxes_in = bboxes[mask.expand_as(bboxes)].view(-1, 4)

        if len(boxes_in) == 0:
            return image, bboxes, labels

        box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(boxes_in)
        boxes_in = boxes_in + box_shift
        labels_in = labels[mask.view(-1)]
        return after_shift_image, boxes_in, labels_in


class RandomCrop(object):

    def __call__(self, image, bboxes, labels):
        center = (bboxes[:, 2:] + bboxes[:, :2]) / 2
        height, width, c = image.shape
        h = random.uniform(0.6 * height, height)
        w = random.uniform(0.6 * width, width)
        offset_x = random.uniform(0, width - w)
        offset_y = random.uniform(0, height - h)
        offset_x, offset_y, h, w = int(offset_x), int(offset_y), int(h), int(w)

        center = center - torch.FloatTensor([[offset_x, offset_y]]).expand_as(center)
        mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
        mask = (mask1 & mask2).view(-1, 1)

        boxes_in = bboxes[mask.expand_as(bboxes)].view(-1, 4)
        if (len(boxes_in) == 0):
            return Image.fromarray(image), bboxes, labels

        box_shift = torch.FloatTensor([[offset_x, offset_y, offset_x, offset_y]]).expand_as(boxes_in)

        boxes_in = boxes_in - box_shift
        boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
        boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
        boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
        boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

        labels_in = labels[mask.view(-1)]
        img_croped = image[offset_y:offset_y + h, offset_x:offset_x + w, :]
        return Image.fromarray(img_croped), boxes_in, labels_in


class Resize(object):
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image, bboxes, labels):
        w, h = image.size
        bboxes /= torch.tensor([w, h, w, h]).expand_as(bboxes)
        image = F.resize(image, (self.image_size, self.image_size))
        return image, bboxes, labels


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, bbox, labels):
        image = F.to_tensor(image).contiguous()
        return image, bbox, labels


class Normalization(object):
    """对图像标准化处理,该方法应放在ToTensor后"""
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image, bbox, labels):
        image = self.normalize(image)
        return image, bbox, labels
