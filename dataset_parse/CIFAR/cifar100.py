import os
import pickle
import cv2


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


if __name__ == '__main__':
    # 数据集地址
    data_path = "/Users/enzo/Documents/GitHub/dataset/CIFAR/cifar-100-python"

    # 从 'batches.meta' 文件中 获取10个类别
    meta_file = os.path.join(data_path, 'meta')
    meta_dict = unpickle(meta_file)
    coarse_label_names = [label.decode() for label in meta_dict[b'coarse_label_names']]  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fine_label_name = [label.decode() for label in meta_dict[b'fine_label_names']]

    # 图片存储地址
    save_path = "/Users/enzo/Documents/GitHub/dataset/CIFAR/cifar-100-images"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for file in ["test"]:
        file_name = os.path.join(data_path, file)
        data_dict = unpickle(file_name)

        images = data_dict[b'data']
        # coarse_label = data_dict[b'coarse_labels']
        fine_label = data_dict[b'fine_labels']

        save_file = os.path.join(save_path, file)
        if not os.path.exists(save_file):
            os.mkdir(save_file)

        for index, image in enumerate(images):
            image = image.reshape(-1, 1024).reshape(-1, 32, 32).transpose(1, 2, 0)
            image_name = str(index) + '_' + fine_label_name[fine_label[index]] + '.png'
            cv2.imwrite(os.path.join(save_file, image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


