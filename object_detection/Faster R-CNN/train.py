from torch.utils.data import Dataset
import os
import torch
from dataset import VOCDataSet
from faster_rcnn_model import FasterRCNN
import network_files.backbone as backbone
from network_files.preprocess import Resize_and_Padding
from network_files.rpn import AnchorsGenerator, RPNHead, RegionProposalNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
voc_root = '/Users/manmi/Documents/GitHub/cv_project/draft/VOC2012'

# --------------------- hyper-parameters ---------------------
batch_size = 8
num_classes = 21
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
rpn_fg_iou_thresh = 0.7   # ground truth 与 anchor 的 iou 大于 0.7,  记为正样本
rpn_bg_iou_thresh = 0.3   # ground truth 与 anchor 的 iou 小于 0.3,  记为负样本，  (0.3, 0.7)之间的舍去
rpn_batch_size_per_image = 256  # 计算损失时，采用正负样本的总个数
rpn_positive_fraction = 0.5  # 正样本占用于计算损失的所有样本的比例
rpn_pre_nms_top_n_train = 2000  # 进行 nms 处理之前， 保留的目标个数
rpn_post_nms_top_n_train = 2000   # nms 处理之后， 所剩余的目标个数，也就是 rpn输出的 proposal 的个数
rpn_pre_nms_top_n_test = 1000
rpn_post_nms_top_n_test = 1000
rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
rpn_nms_thresh = 0.7   # nms 阈值
rpn_score_thresh = 0.0


# ====================================================================
# ====================== DataSet and Dataloader ======================
# ====================================================================
train_dataset = VOCDataSet(voc_root)
valid_dataset = VOCDataSet(voc_root, train_set=False)


nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])   # 8

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=0,
                                           collate_fn=train_dataset.collate_fn)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=nw,
                                           collate_fn=valid_dataset.collate_fn)

# ====================================================================
# ======================== Faster R-CNN model ========================
# ====================================================================


resize_and_padding = Resize_and_Padding(min_size=800, max_size=1333)
backbone = backbone.resnet18_backbone()

rpn_anchor_generator = AnchorsGenerator(anchor_sizes, aspect_ratios)
rpn_head = RPNHead(backbone.out_channels, len(aspect_ratios[0])*len(anchor_sizes))
rpn = RegionProposalNetwork(rpn_anchor_generator, rpn_head,
                            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                            rpn_batch_size_per_image, rpn_positive_fraction,
                            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
                            score_thresh=rpn_score_thresh)

model = FasterRCNN(backbone, rpn, resize_and_padding)


# ====================================================================
# ============================== Train ===============================
# ====================================================================
model.to(device)
for images, targets in train_loader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    proposals, proposal_losses = model(images, targets)

    break
