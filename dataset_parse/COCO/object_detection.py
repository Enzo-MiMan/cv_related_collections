import json
import os.path
import os
import torch
import numpy as np
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def read_file(anno_root, train_val):
    annotation_file = os.path.join(anno_root, train_val)
    assert os.path.exists(annotation_file), '{} not exists'.format(annotation_file)
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    categories = data['categories']
    images = data['images']
    annotations = data['annotations']
    print('{} has {} images'.format(train_val.split('.')[0], len(images)))
    return categories, images, annotations


anno_root = '/Users/enzo/Documents/GitHub/dataset/COCO2017/annotations'

categories, train_images, train_annotations = read_file(anno_root, "instances_train2017.json")
_, val_images, val_annotations = read_file(anno_root, "instances_val2017.json")


save_file =
with open(save_file, 'w') as f:
    for image in train_images:
        image_anno = {}
        image_anno['file_name'] = image['file_name']
        image_anno['hw'] = [image['height'], image['width']]
        id = image['id']

        bboxes, labels = [], []
        for anno in train_annotations:
            if anno['image_id'] == id:
                bboxes.append(anno['bbox'])
                labels.append(anno['category_id'])
        image_anno['bboxes'] = bboxes
        image_anno['labels'] = labels

        f.write(image_anno)
        torch.save()






