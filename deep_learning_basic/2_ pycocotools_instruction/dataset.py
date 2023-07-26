from pycocotools.coco import COCO
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import random

# ============================================================================================
# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  getAnnIds  - Get ann ids that satisfy given filter conditions. 用 category id / image id 获取 annotation id
#  getCatIds  - Get cat ids that satisfy given filter conditions. 用 catNms （类别名称） / supNms（大类的类别名称）等获取 category id
#  getImgIds  - Get img ids that satisfy given filter conditions. 用 category id 获取 image id
#  loadAnns   - Load anns with the specified ids.  用 annotation id 获取 annotation 信息
#  loadCats   - Load cats with the specified ids.  用 category id 获取 category 信息
#  loadImgs   - Load imgs with the specified ids.  用 image id 获取 image 信息
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# ============================================================================================


# 定义数据集路径
root = Path("/Users/enzo/Documents/GitHub/dataset/COCO2017")
train_img_file = os.path.join(root, 'train2017')
val_img_file = os.path.join(root, 'val2017')
train_ann_file = os.path.join(root, 'annotations/instances_train2017.json')
val_ann_file = os.path.join(root, 'annotations/instances_val2017.json')

assert os.path.exists(train_ann_file), f'provided COCO path {root} does not exist'
assert os.path.exists(val_ann_file), f'provided COCO path {root} does not exist'


coco = COCO(train_ann_file)

print('\n', '*'*40, '\n')



# ------------ 1、getAnnIds  ------------

"""
getAnnIds(imgIds=[], catIds=[], areaRng=[], iscrowd=None)
Get ann ids that satisfy given filter conditions. default skips that filter
:param imgIds  (int array)     : get anns for given imgs
       catIds  (int array)     : get anns for given cats
       areaRng (float array)   : get anns for given area range (e.g. [0 inf])
       iscrowd (boolean)       : get anns for given crowd label (False or True)
:return: ids (int array)       : integer array of ann ids
"""

ids = coco.getAnnIds(catIds=[18])
print(len(ids))
# 5508

# ------------ 2、getCatIds ------------

"""
getCatIds(catNms=[], supNms=[], catIds=[])
filtering parameters. default skips that filter.
:param catNms (str array)  : get cats for given cat names
:param supNms (str array)  : get cats for given supercategory names
:param catIds (int array)  : get cats for given cat ids
:return: ids (int array)   : integer array of cat ids
"""

id = coco.getCatIds(catNms=['dog'])
ids = coco.getCatIds(supNms=['animal'])
print(id)
print(ids)
# [18]
# [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

# ------------ 3、getImgIds ------------

'''
getImgIds(imgIds=[], catIds=[])
Get img ids that satisfy given filter conditions.
:param imgIds (int array) : get imgs for given ids
:param catIds (int array) : get imgs with all given cats
:return: ids (int array)  : integer array of img ids
'''

ids = coco.getImgIds(catIds=[18])
print(len(ids))
# 4385


# ------------ 4、loadAnns ------------

"""
loadAnns(ids=[])
Load anns with the specified ids.
:param ids (int array)       : integer ids specifying anns
:return: anns (object array) : loaded ann objects
"""

anns = coco.loadAnns(ids=[1727])
print(anns[0].keys())
# dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])


# ------------ 5、loadCats ------------

"""
loadCats(ids=[])
Load cats with the specified ids.
:param ids (int array)       : integer ids specifying cats
:return: cats (object array) : loaded cat objects
"""

cats = coco.loadCats(ids=[18])
print(cats)
# [{'supercategory': 'animal', 'id': 18, 'name': 'dog'}]


# ------------ 6、loadImgs ------------

"""
loadImgs(ids=[])
Load anns with the specified ids.
:param ids (int array)       : integer ids specifying img
:return: imgs (object array) : loaded img objects
"""

imgs = coco.loadImgs(ids=[98304])
print(imgs)
# [{'license': 1,
#   'file_name': '000000098304.jpg',
#   'coco_url': 'http://images.cocodataset.org/train2017/000000098304.jpg',
#   'height': 424,
#   'width': 640,
#   'date_captured': '2013-11-21 23:06:41',
#   'flickr_url': 'http://farm6.staticflickr.com/5062/5896644212_a326e96ea9_z.jpg',
#   'id': 98304}]



# ------------ 7、showAnns ------------

"""
showAnns(anns)
Display the specified annotations.
:param anns (array of object): annotations to display
:return: None
"""

img_file = os.path.join(train_img_file, '000000098304.jpg')
img = cv2.imread(img_file)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

anns = coco.loadAnns(coco.getAnnIds(imgIds=[98304]))
coco.showAnns(anns)
plt.show()


# ------------ 7、loadRes ------------

"""
loadRes(resFile)
Load result file and return a result api object.
:param   resFile (str)     : file name of result file
:return: res (obj)         : result api object
"""


resFile = "results.json"
cocoRes = coco.loadRes(resFile)


