import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class yoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        device = pred_tensor.device
        # ================== 数据准备 ==================
        # 有object 的 mask：coo_mask
        # 无object 的 mask：noo_mask
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:, :, :, 4] > 0
        noo_mask = target_tensor[:, :, :, 4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        # ================== 1、object 置信度误差 ==================
        # 1.1 compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1, 30)
        noo_target = target_tensor[noo_mask].view(-1, 30)
        noo_pred_mask = torch.ByteTensor(noo_pred.size()).to(device)
        noo_pred_mask.zero_()
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        noo_pred_c = noo_pred[noo_pred_mask]  # noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum')

        # 1.2 compute contain obj loss
        # 从 预测结果中 取出有 objebt 的 置信度数据
        coo_pred = pred_tensor[coo_mask].view(-1, 30)
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:, 10:]  # conditional probability for 20 classes

        # 从 target中 取出有 objebt 的 置信度数据
        coo_target = target_tensor[coo_mask].view(-1, 30)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        coo_response_mask = torch.ByteTensor(box_target.size()).to(device)
        coo_response_mask.zero_()
        coo_not_response_mask = torch.ByteTensor(box_target.size()).to(device)
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).to(device)
        for i in range(0, box_target.shape[0], 2):  # choose the best iou box
            box1 = box_pred[i:i + 2]
            box1_xyxy = torch.FloatTensor(box1.size())
            box1_xyxy[:, :2] = box1[:, :2] / 14. - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] / 14. + 0.5 * box1[:, 2:4]

            box2 = box_target[i].view(-1, 5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] / 14. - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / 14. + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            max_iou, max_index = iou.max(0)
            max_iou, max_index = max_iou.to(device), max_index.to(device)

            coo_response_mask[i + max_index] = 1
            coo_not_response_mask[i + 1 - max_index] = 1

            box_target_iou[i + max_index] = max_iou

        # 1.负责检测的box 的 predict 和 target 的 （1）位置坐标 （2）object 置信度 、 IOU
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)

        # 坐标误差，xy误差 + wh误差
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum') \
                   + F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]),torch.sqrt(box_target_response[:, 2:4]), reduction='sum')

        # 包含object的grid cell， 计算 object 置信度误差
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum')

        # 负责检测的grid cell 中不负责检测的bboox，计算 object 置信度误差
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], reduction='sum')

        # 3.class loss
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        return (self.l_coord * loc_loss + 2 * contain_loss + not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N


