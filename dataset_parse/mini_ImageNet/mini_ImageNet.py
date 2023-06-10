import os
import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def read_csv_classes(data_path, file_name):
    data = pd.read_csv(os.path.join(data_path, file_name))
    label_set = set(data["label"].drop_duplicates().values)
    print("{} have {} images and {} classes.".format(file_name, data.shape[0], len(label_set)))
    return data, label_set


def split_dataset(image_dir, data, labels, rate=0.8):
    # split data on every classes
    num_every_classes = []
    split_train_data = []
    split_val_data = []

    for label in labels:
        class_data = data[data["label"] == label]
        num_every_classes.append(class_data.shape[0])

        # shuffle
        shuffled_data = class_data.sample(frac=1, random_state=1)
        num_train_sample = int(class_data.shape[0] * rate)
        split_train_data.append(shuffled_data[:num_train_sample])
        split_val_data.append(shuffled_data[num_train_sample:])

        # # 可视化每个类别的 9张图像
        # for i in range(9):
        #     img_name, img_label = shuffled_data.iloc[i].values
        #     img = Image.open(os.path.join(image_dir, img_name))
        #     plt.subplot(3, 3, i+1)
        #     plt.imshow(img)
        #     plt.title("class: " + classes_label[img_label])
        # plt.show()

    # plot classes distribution
    plot_flag = False
    if plot_flag:
        plt.bar(range(1, 101), num_every_classes, align='center')
        plt.show()

    # 合并 训练集 和 验证集的 所有类别的数据
    new_train_data = pd.concat(split_train_data, axis=0)
    new_val_data = pd.concat(split_val_data, axis=0)

    return new_train_data, new_val_data


if __name__ == '__main__':
    # 数据集根目录
    data_path = "/Users/enzo/Documents/GitHub/dataset/mini-imagenet"

    # 从 json文件中获取数据集的 1000个类别
    classes_file = os.path.join(data_path, "ImageNet_classes.json")
    with open(classes_file, 'r') as f:
        label_dict = json.load(f)
    label_dict = dict([(index, name) for index, name in label_dict.values()])

    # 从 训练集、验证集、测试集中 解析出 数据 和 标签
    train_data, train_label_set = read_csv_classes(data_path, "train.csv")
    val_data, val_label_set = read_csv_classes(data_path, "val.csv")
    test_data, test_label_set = read_csv_classes(data_path, "test.csv")

    # 读取所有图像： 60000张
    image_dir = os.path.join(data_path, "images")
    images_list = [image_name for image_name in os.listdir(image_dir) if image_name.endswith(".jpg")]
    print("find {} images in dataset.".format(len(images_list)))

    # 合并 训练集、验证集、测试集 中所有的标签
    labels = list(train_label_set | val_label_set | test_label_set)
    labels.sort()
    print("all classes: {}".format(len(labels)))

    # 将 mini imagenet 的 100个 标签保存为 classes_name.json
    classes_label = dict([(label, label_dict[label]) for label in labels])
    json_str = json.dumps(classes_label, indent=4)
    with open(os.path.join(data_path, 'mini_ImageNet_classes.json'), 'w') as json_file:
        json_file.write(json_str)

    # 合并训练集、验证集、测试集的 data
    data = pd.concat([train_data, val_data, test_data], axis=0)
    print("total data shape: {}".format(data.shape))

    # 重新按照每个类别 8:2 的比例，划分训练集和验证集
    new_train_data, new_val_data = split_dataset(image_dir, data, labels)

    # save new csv data
    new_train_data.to_csv(os.path.join(data_path, "new_train.csv"), index=False)
    new_val_data.to_csv(os.path.join(data_path, "new_val.csv"), index=False)
