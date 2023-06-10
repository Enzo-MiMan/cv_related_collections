import xml.etree.ElementTree as ET
import os


# voc的20个类别
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def convert(size, bbox):
    x = (bbox[0] + bbox[1]) / 2.0
    y = (bbox[2] + bbox[3]) / 2.0
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    x = x / size[0]
    w = w / size[0]
    y = y / size[1]
    h = h / size[1]
    return (x, y, w, h)


def convert_annotation(xml_file, save_file):

    # 保存yolo格式 的label 的 .txt 文件地址
    save_file = open(save_file, 'w')

    tree = ET.parse(xml_file)
    size = tree.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in tree.findall('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls) + 1   # 类别索引从1开始，类别0是背景
        bbox = obj.find('bndbox')
        b = (float(bbox.find('xmin').text),
             float(bbox.find('xmax').text),
             float(bbox.find('ymin').text),
             float(bbox.find('ymax').text))
        bb = convert((w, h), b)
        save_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    save_file.close()


if __name__ == "__main__":
    # 数据集根目录地址
    data_root = "/Users/enzo/Documents/GitHub/dataset/VOCdevkit/VOC2007"

    # 标注文件地址
    annotation = os.path.join(data_root, 'Annotations')

    # yolo格式的文件保存地址
    save_root = './labels'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for train_val in ["train", "val"]:
        if not os.path.exists(os.path.join(save_root, train_val)):
            os.makedirs(os.path.join(save_root, train_val))

        # 数据集划分的 .txt 文件地址
        txt_file = os.path.join(data_root, 'ImageSets/Main', train_val+'.txt')

        with open(txt_file, 'r') as f:
            lines = f.readlines()
        file_names = [line.strip() for line in lines if len(line.strip())>0]

        for file_name in file_names:
            xml_file = os.path.join(annotation, file_name+'.xml')
            save_file = os.path.join(save_root, train_val, file_name+'.txt')

            convert_annotation(xml_file, save_file)
