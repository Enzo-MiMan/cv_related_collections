import numpy as np


def nms(boxes, scores, threshold):
    """
    对预测框进行NMS操作
    Args:
        boxes: array, 预测框, (n, 4)
        scores: array, 预测框置信度, (n, )
        threshold: float, 阈值
    Returns:
        保留的预测框索引
    """
    # 获取预测框左上角和右下角坐标
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # 计算预测框面积
    areas = (x2 - x1) * (y2 - y1)
    # 降序排序
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        # 取出当前置信度最高的预测框
        i = order[0]
        keep.append(i)
        # 获取当前预测框与其他预测框的交叠部分左上角和右上角坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算交叠部分面积
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        # 计算交叠部分占预测框面积的比例（IoU）
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留小于阈值的预测框
        inds = np.where(iou <= threshold)[0]
        # 保留下来的索引
        order = order[inds + 1]
    return keep
