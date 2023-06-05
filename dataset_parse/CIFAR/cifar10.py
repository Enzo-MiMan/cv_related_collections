import os
import pickle
import cv2


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


if __name__ == '__main__':
    # 数据集地址
    data_path = "/Users/enzo/Documents/GitHub/dataset/CIFAR/cifar-10-batches-py"

    # 5个 train batch 和 1个 test batch 的文件夹名称
    batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

    # 从 'batches.meta' 文件中 获取10个类别
    meta_file = os.path.join(data_path, 'batches.meta')
    meta_dict = unpickle(meta_file)
    label_name = [label.decode() for label in meta_dict[b'label_names']]  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 图片存储地址
    save_path = "/Users/enzo/Documents/GitHub/dataset/CIFAR/cifar-10-images"


    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for batch in batches:
        batch_file = os.path.join(data_path, batch)
        batch_dict = unpickle(batch_file)

        images = batch_dict[b'data']
        labels = batch_dict[b'labels']

        save_file = os.path.join(save_path, batch)
        if not os.path.exists(save_file):
            os.mkdir(save_file)

        for index, image in enumerate(images):
            image = image.reshape(-1, 1024).reshape(-1, 32, 32).transpose(1, 2, 0)
            image_name = str(index) + '_' + label_name[labels[index]] + '.png'
            cv2.imwrite(os.path.join(save_file, image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


