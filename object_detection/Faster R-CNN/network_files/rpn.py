import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision
import network_files.det_utils as det_utils
import torch
from typing import Tuple
from torch import Tensor


class RegionProposalNetwork(torch.nn.Module):
    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,    # 0.7, 0.3
                 batch_size_per_image, positive_fraction,   # 256   0.5
                 pre_nms_top_n, post_nms_top_n,
                 nms_thresh, score_thresh=0.0):   # 2000, 2000

        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder()

        # use during training
        # 计算anchors与真实bbox的iou
        # self.box_similarity = box_iou

        # self.proposal_matcher = det_utils.Matcher(
        #     fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
        #     bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
        #     allow_low_quality_matches=True
        # )

        # self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
        #     batch_size_per_image, positive_fraction  # 256, 0.5
        # )

        # use during testing
        self.pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1.

    def forward(self, images, features, targets=None):

        features = features['0']
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)
        num_images = len(anchors)
        num_anchors = objectness.shape[1] * objectness.shape[2] * objectness.shape[3]

        N, A, H, W = objectness.shape
        objectness = permute_and_flatten(objectness, N, 1, H, W)  # shape : (batch_size * Anchor * Height * Width, 1)
        pred_bbox_deltas = permute_and_flatten(pred_bbox_deltas, N, 4, H, W)

        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        # 筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes)

    def filter_proposals(self, proposals, objectness, image_shapes):
        # # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]

        num_images = proposals.shape[0]
        device = proposals.device

        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # select top_n boxes before applying nms
        # self.pre_nms_top_n={'training': 2000, 'testing': 1000}
        if self.training:
            pre_nms_top_n = self.pre_nms_top_n['training']
        else:
            pre_nms_top_n = self.pre_nms_top_n['testing']

        _, top_n_idx = objectness.topk(pre_nms_top_n, dim=1)

        # 排前 pre_nms_top_n 的预测概率
        batch_idx = torch.arange(num_images, device=device).reshape(-1, 1)
        objectness = objectness[batch_idx, top_n_idx]
        # 预测概率排前pre_nms_top_n的 proposal 的坐标信息
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        for boxes, scores, img_shape in zip(proposals, objectness_prob, image_shapes):
            # 调整预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = clip_boxes_to_image(boxes, img_shape)

            # 返回boxes满足宽，高都大于min_size的索引
            keep = remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 移除小概率boxes，参考下面这个链接
            # https://github.com/pytorch/vision/pull/3205
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]  # ge: >=
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def _get_top_n_idx(self, objectness, num_anchors):
        # type: (Tensor, List[int]) -> Tensor
        if self.training:
            pre_nms_top_n = self.pre_nms_top_n['training']
        else:
            pre_nms_top_n = self.pre_nms_top_n['testing']

        _, top_n_idx = num_anchors.topk(pre_nms_top_n, dim=1)
        return top_n_idx

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        """
        计算每个anchors最匹配的gt，并划分为正样本，背景以及废弃的样本
        Args：
            anchors: (List[Tensor])
            targets: (List[Dict[Tensor])
        Returns:
            labels: 标记anchors归属类别（1, 0, -1分别对应正样本，背景，废弃的样本）
                    注意，在RPN中只有前景和背景，所有正样本的类别都是1，0代表背景
            matched_gt_boxes：与anchors匹配的gt
        """
        labels = []
        matched_gt_boxes = []
        # 遍历每张图像的anchors和targets
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            # 计算anchors与真实bbox的iou信息
            # set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
            match_quality_matrix = box_iou(gt_boxes, anchors_per_image)
            # 计算每个anchors与gt匹配iou最大的索引（如果iou<0.3索引置为-1，0.3<iou<0.7索引为-2）
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

            # 记录所有anchors匹配后的标签(正样本处标记为1，负样本处标记为0，丢弃样本处标记为-2)
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # background (negative examples)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
            labels_per_image[bg_indices] = 0.0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
            labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes


class AnchorsGenerator(torch.nn.Module):
    def __init__(self, sizes, aspect_ratios):
        # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        super(AnchorsGenerator, self).__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def forward(self, image_list, feature_maps):
        feature_map_size = feature_maps.shape[-2:]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps.dtype, feature_maps.device

        # [原图高/特征图高， 原图宽/特征图宽]
        strides = [torch.tensor(image_size[0] // feature_map_size[0], dtype=torch.int64, device=device),
                   torch.tensor(image_size[1] // feature_map_size[1], dtype=torch.int64, device=device)]

        # 获得 15个 anchor 的坐标, 以anchor中心点为(0, 0)点，  shape=(15, 4)
        cell_anchors = [self.generate_anchors(sizes, aspect_ratios, dtype, device) for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)]
        self.cell_anchors = torch.concat(cell_anchors, dim=0)

        # 将 anchor 按照 strides 映射回 原图尺寸
        anchors_in_sourceimg_scale = self.grid_anchors(feature_map_size, strides)

        # 将同样的 anchors_in_sourceimg_scale 复制出来 batch_size 个
        anchors = [anchors_in_sourceimg_scale for i in range(feature_maps.shape[0])]
        return anchors

    def generate_anchors(self, scales, aspect_ratios, dtype, device):
        # # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()  # round 四舍五入

    def grid_anchors(self, feature_map_size, strides):
        # # type: (torch.Size([int, int]), List[Tensor, Tensor]) -> List[Tensor]
        cell_anchors = self.cell_anchors

        grid_height, grid_width = feature_map_size
        stride_height, stride_width = strides
        device = cell_anchors[0].device

        shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
        shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

        shift_y, shift_x = torch.meshgrid([shifts_y, shifts_x], indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
        shifts_anchor = shifts.view(-1, 1, 4) + cell_anchors.view(1, -1, 4)
        return shifts_anchor.reshape(-1, 4)  # Tensor(all_num_anchors, 4)  其中  all_num_anchors = height * width * 15


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算s（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # #  type: (Tensor) -> Tuple[Tensor, Tensor]
        t = F.relu(self.conv(x))
        return self.cls_logits(t), self.bbox_pred(t)


def permute_and_flatten(layer, N, C, H, W):
    """
    调整tensor顺序，并进行reshape
    Args:
        layer: 预测特征层上预测的 目标概率 或 bboxes regression参数 -- RPNHead 的输出
               box_cls shape : [batch_size, anchors_num_per_position, height, width]
               bboxes regression shape:  [batch_size, anchors_num_per_position * 4), height, width]
        N: batch_size
        A: anchors_num_per_position
        C: classes_num or 4(bbox coordinate)
        H: height
        W: width
    Returns:
        layer: 调整tensor顺序，并reshape后的结果[N, -1, C]
    """

    layer = layer.view(N, -1, C,  H, W)
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    layer = layer.reshape(-1, C)
    return layer   # shape : (batch_size * Anchor * Height * Width, C)


def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int, int]) -> Tensor

    boxes_x = boxes[:, 0::2]  # x1, x2
    boxes_y = boxes[:, 1::2]  # y1, y2
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)   # 限制x坐标范围在[0,width]之间
    boxes_y = boxes_y.clamp(min=0, max=height)  # 限制y坐标范围在[0,height]之间

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=2)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes, min_size):
    # type: (Tensor, float) -> Tensor
    ws = boxes[:, 2] - boxes[:, 0]   # 预测boxes的宽和高
    hs = boxes[:, 3] - boxes[:, 1]  # 预测boxes的宽和高
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    keep = torch.where(keep)[0]
    return keep


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
