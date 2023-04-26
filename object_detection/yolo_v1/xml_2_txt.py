from lxml import etree
import os


class_dict = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}


def parse_xml(xml_path, class_dict):
    """ Parse a PASCAL VOC xml file """
    tree_root = etree.parse(xml_path)
    data_dict = []

    for obj in tree_root.findall('object'):
        obj_info = {}
        #  解析出 difficult， 并丢弃 difficult==1 的数据
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            continue
        # 解析 object 的 class 和 bbox坐标
        class_name = obj.find('name').text
        obj_info['name'] = class_dict[class_name]
        bbox = obj.find('bndbox')
        obj_info['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        data_dict.append(obj_info)

    return data_dict


# 数据集地址
data_root = '/Users/enzo/Documents/GitHub/dataset/VOCdevkit/VOC2012'
Annotations = os.path.join(data_root, 'Annotations')
image_path = os.path.join(data_root, 'JPEGImages')
train_txt = os.path.join(data_root, 'ImageSets/Main/train.txt')
val_txt = os.path.join(data_root, 'ImageSets/Main/val.txt')
# 检查数据集地址
assert os.path.exists(Annotations), 'Annotations file not exist.'
assert os.path.exists(image_path), 'JPEGImages file not exist.'
assert os.path.exists(train_txt), 'train.txt file not exist.'
assert os.path.exists(val_txt), 'val.txt file not exist.'

# 创建保存地址
save_dir = './my_yolo_dataset'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_train = os.path.join(save_dir, 'train_label_bbox.txt')
save_val = os.path.join(save_dir, 'val_label_bbox.txt')

# 读取所有 .xml 文件名
xml_files = os.listdir(Annotations)
with open(train_txt) as f:
    train_lines = [line.strip() for line in f.readlines() if len(line.strip())>0]
with open(val_txt) as f:
    val_lines = [line.strip() for line in f.readlines() if len(line.strip())>0]

count = 0
for train_val in ['train', 'val']:

    if train_val == 'train':
        lines = train_lines
        save_file = save_train
    else:
        lines = val_lines
        save_file = save_val

    with open(save_file, 'w') as f:
        for name in lines:
            # 写入图像 .jpg 图像名
            image_path = name + '.jpg'

            f.write(image_path)
            # 解析xml文件, 获取其中object 的 class name 和 bbox坐标
            data = parse_xml(os.path.join(Annotations, name + '.xml'), class_dict)
            if len(data) == 0:
                continue
            for obj in data:
                class_name = obj['name']
                bbox = obj['bbox']
                # 写入每个object 的 class name 和 bbox坐标
                f.write(' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name))
            f.write('\n')

