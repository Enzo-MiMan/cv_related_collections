import torch
import torch.nn as nn
import torch.distributed as dist


def criterion(predict, target):
    losses = {}
    for name, x in predict.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    conf_mat = ConfusionMatrix(num_classes)
    with torch.no_grad():

        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            predict = model(image)
            predict = predict['out']

            conf_mat.update(target.flatten(), predict.argmax(1).flatten())

        conf_mat.reduce_from_all_processes()
    return conf_mat


def train_one_epoch(model, optimizer, data_loader, device, lr_scheduler, scaler=None):
    model.train()

    for image, target in data_loader:
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            predict = model(image)
            loss = criterion(predict, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

    return loss, optimizer.param_groups[0]["lr"]


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):

    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中，学习率因子（learning rate factor）： warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后， warmup过程中，学习率因子（learning rate factor）：1 -> 0
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, predict):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + predict[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iou

    def reduce_from_all_processes(self):
        '''
        torch.distributed.is_available()： 函数用于检测当前 PyTorch 是否支持分布式训练。
            如果返回 True，表示当前安装的 PyTorch 支持分布式训练；
            如果返回 False，表示当前安装的 PyTorch 不支持分布式训练或未安装分布式训练相关的扩展库。

        torch.distributed.is_initialized()： 函数用于检测当前进程是否已经初始化了分布式训练环境。
            如果返回True，表示当前进程已经完成分布式环境的初始化；
            如果返回False，表示当前进程还未完成分布式环境的初始化，或当前没有初始化分布式环境。
        '''
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

