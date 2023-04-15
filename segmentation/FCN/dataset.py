import os
import torch.utils.data as data
import numpy as np
import torch
import transforms
from PIL import Image
import matplotlib.pyplot as plt


class VOCSegmentation(data.Dataset):
    def __init__(self, img_path, gt_path, txt_file, train_val='train', base_size=520, crop_size=480, flip_prob=0.5):
        super(VOCSegmentation, self).__init__()

        if train_val == 'train':
            with open(txt_file, 'r') as f:
                data = [data.strip() for data in f.readlines() if len(data.strip()) > 0]
            self.transforms = transforms.Compose([transforms.RandomResize(int(base_size*0.5), int(base_size*2)),
                                                  transforms.RandomHorizontalFlip(flip_prob),
                                                  transforms.RandomCrop(crop_size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                  ])
        else:
            with open(txt_file, 'r') as f:
                data = [data.strip() for data in f.readlines() if len(data.strip()) > 0]
            self.transforms = transforms.Compose([transforms.RandomResize(base_size, base_size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.img_files = [os.path.join(img_path, i + '.jpg') for i in data]
        self.gt_files = [os.path.join(gt_path, i + '.png') for i in data]

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        target = Image.open(self.gt_files[index])
        img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs







