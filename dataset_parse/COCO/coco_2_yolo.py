import os
import json
from tqdm import tqdm
import argparse


def convert(size, box):
    '''
    size: 图片的宽和高(w,h)
    box格式: x,y,w,h
    返回值：x_center/image_width y_center/image_height width/image_width height/image_height
    '''

    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default='test.json',
                        type=str, help="coco file path")
    parser.add_argument('--save_dir', default='labels', type=str,
                        help="where to save .txt labels")
    arg = parser.parse_args()

    data = json.load(open(arg.json_file, 'r'))

    # 如果存放txt文件夹不存在，则创建
    if not os.path.exists(arg.save_dir):
        os.makedirs(arg.save_dir)

    id_map = {}

    # 解析目标类别，也就是 categories 字段，并将类别写入文件 classes.txt 中
    with open(os.path.join(arg.save_dir, 'classes.txt'), 'w') as f:
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i

    for img in tqdm(data['images']):

        # 解析 images 字段，分别取出图片文件名、图片的宽和高、图片id
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)

        # txt文件名，与对应图片名只有后缀名不一样
        txt_name = head + ".txt"
        f_txt = open(os.path.join(arg.save_dir, txt_name), 'w')

        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])

                # 写入txt，共5个字段
                f_txt.write("%s %s %s %s %s\n" % (
                    id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))

        f_txt.close()