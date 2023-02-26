from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree
from torchvision.transforms import functional as F
import random


class VOCDataSet(Dataset):
    def __init__(self, voc_root, train_set=True):
        self.root = voc_root
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.annotations_root = os.path.join(self.root, 'Annotations')

        # read train.txt or val.txt file
        if train_set:
            txt_list = os.path.join(self.root, 'ImageSets', 'Main', 'train.txt')
        else:
            txt_list = os.path.join(self.root, 'ImageSets', 'Main', 'val.txt')
        with open(txt_list) as f:
            self.xml_list = [os.path.join(self.annotations_root, line.strip()+'.xml') for line in f.readlines()]

        # read class_indict
        with open('./pascal_voc_classes.json', 'r') as json_file:
            self.class_dict = json.load(json_file)

        if train_set:
            self.trans = Compose([ToTensor(),
                                  RandomHorizontalFlip(0.5),
                                  Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.trans = Compose([ToTensor(),
                                  Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        xml_path = self.xml_list[idx]
        filename, target = self.parse_xml(xml_path)
        target['image_id'] = torch.tensor([idx])

        img_path = os.path.join(self.img_root, filename)
        image = Image.open(img_path)

        image, target = self.trans(image, target)
        return image, target

    def parse_xml(self, xml_path):
        data_dict = {}
        tree_root = etree.parse(xml_path)
        # file name
        filename = tree_root.find("filename").text

        class_list = []
        coord_list = []
        area_list = []
        for object in tree_root.findall("object"):
            # class
            obj_class = self.class_dict[object.find("name").text]
            class_list.append(obj_class)
            # bounding box
            bbox = object.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            coord_list.append([xmin, ymin, xmax, ymax])
            # area
            area = (ymax - ymin) * (xmax - xmin)
            area_list.append(area)

        data_dict['labels'] = torch.as_tensor(class_list, dtype=torch.int64)
        data_dict['boxes'] = torch.as_tensor(coord_list, dtype=torch.float32)
        data_dict['area'] = torch.as_tensor(area_list, dtype=torch.float32)
        return filename, data_dict

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prop=0.5):
        self.prop = prop

    def __call__(self, image, target):
        if random.random() < self.prop:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target['boxes']
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target['boxes'] = bbox
        return image, target


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, image, target):
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return image, target


# if __name__ == '__main__':
#     voc_root = '/Users/manmi/Documents/GitHub/cv_project/draft/VOC2012'
#     train_dataset = VOCDataSet(voc_root)
#     valid_dataset = VOCDataSet(voc_root, train_set=False)
#
#     batch_size = 8
#     nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
#     print('nw:', nw)
#
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=batch_size,
#                                                shuffle=True,
#                                                pin_memory=True,
#                                                num_workers=nw,
#                                                collate_fn=train_dataset.collate_fn)
#
#     valid_loader = torch.utils.data.DataLoader(valid_dataset,
#                                                batch_size=1,
#                                                shuffle=False,
#                                                pin_memory=True,
#                                                num_workers=nw,
#                                                collate_fn=valid_dataset.collate_fn)
#
#     image, target = train_dataset[0]
#     print('target:')
#     for k, v in target.items():
#         print(f'   {k}:  {v}')
#     print('=================================')
#     print('image : \n', image)
