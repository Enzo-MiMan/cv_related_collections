import os
import torch
import torch.utils.data as data
import transforms
from PIL import Image


class yolo_Dataset(data.Dataset):
    image_size = 448

    def __init__(self, image_path, txt_file, train=True):
        self.image_path = image_path
        self.train = train
        self.mean = (123, 117, 104)

        self.fnames = []
        self.bboxes = []
        self.labels = []

        with open(txt_file) as f:
            lines = [line.strip() for line in f.readlines() if len(line.strip())>0]

        for line in lines:
            splited = line.split()
            # 获取图像名 .jpg
            self.fnames.append(splited[0])
            # 图像中的 object 的个数
            num_boxes = (len(splited)-1) // 5
            # 获取图像中每个 objct 的bbox坐标 和 类别label
            bbox = []
            label = []
            for i in range(num_boxes):
                x = float(splited[1 + 5 * i])
                y = float(splited[2 + 5 * i])
                x2 = float(splited[3 + 5 * i])
                y2 = float(splited[4 + 5 * i])
                c = int(splited[5+5*i])
                bbox.append([x, y, x2, y2])
                label.append(c)

            self.bboxes.append(torch.tensor(bbox))
            self.labels.append(torch.LongTensor(label))

        if train:
            self.my_trans = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                                transforms.RandomScale(0.8, 1.2, 0.5),
                                                transforms.ColorJitter(),
                                                transforms.RandomShift([0.485, 0.456, 0.406]),
                                                transforms.RandomCrop(),
                                                transforms.Resize(self.image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalization(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
        else:
            self.my_trans = transforms.Compose([transforms.Resize(self.image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalization(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])


    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image = Image.open(os.path.join(self.image_path, fname))
        bboxes = self.bboxes[idx]
        labels = self.labels[idx]

        image, bboxes, labels = self.my_trans(image, bboxes, labels)
        target = self.encoder(bboxes, labels)
        return image, target


    def encoder(self, bboxes, labels):
        grid_num = 7
        target = torch.zeros((grid_num, grid_num, 30))
        cell_size = 1./grid_num
        wh = bboxes[:, 2:] - bboxes[:, :2]
        cxcy = (bboxes[:,2:] + bboxes[:,:2]) / 2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1  #
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1
            xy = ij * cell_size  # 匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target


    def __len__(self):
        return len(self.bboxes)

