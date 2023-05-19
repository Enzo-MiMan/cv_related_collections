import datetime
import argparse
import os
import time
import torch
from dataset import VOCSegmentation
from torch.utils import data
from utils import create_lr_scheduler, train_one_epoch, evaluate
from fcn_model import fcn_resnet50
import shutil


def create_model(num_classes, pretrain=True):
    model = fcn_resnet50(aux=True, num_classes=num_classes)
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'
    if pretrain:
        weights_dict = torch.load('./fcn_resnet50_coco.pth', map_location='cpu')
        if num_classes != 21:
            for k in list(weights_dict.keys()):
                if 'classifier.4' in k:
                    del weights_dict[k]
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print('missing_keys:', missing_keys)
            print('unexpected_keys:', unexpected_keys)
    return model


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes + 1

    if os.path.exists('./results'):
        shutil.rmtree('./results')
        os.mkdir('./results')
    results_file = "results/result_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    # #################### Dataset & DataLoader ##############################

    root = args.data_path
    img_path = os.path.join(root, 'JPEGImages')
    gt_path = os.path.join(root, 'SegmentationClass')
    train_txt = os.path.join(root, 'ImageSets/Segmentation/train.txt')
    val_txt = os.path.join(root, 'ImageSets/Segmentation/val.txt')

    assert os.path.exists(img_path), 'img_path not exists'
    assert os.path.exists(gt_path), 'gt_path not exists'
    assert os.path.exists(train_txt), 'train_txt not exists'
    assert os.path.exists(val_txt), 'val_txt not exists'

    train_dataset = VOCSegmentation(img_path, gt_path, train_txt, train_val='train')
    val_dataset = VOCSegmentation(img_path, gt_path, val_txt, train_val='val')

    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               # num_workers=1,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=2,
                                             # num_workers=1,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)


    # ############################## Module & optimizer & scheduler ##############################

    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [
        {'params': [p for p in model.backbone.parameters() if p.requires_grad]},
        {'params': [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({'params': params, 'lr': args.lr*10})

    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # 学习率更新策略：每个step更新一次 (不是每个epoch 更新一次)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])


    # ###################### Train ######################
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device,
                                        lr_scheduler=lr_scheduler, scaler=scaler)

        conf_mat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(conf_mat)
        print(val_info)

        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" + f"train_loss: {mean_loss:.4f}\n" + f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, "save_weights/model.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():

    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--data-path", default="/Users/enzo/Documents/GitHub/dataset/VOCdevkit/VOC2012", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default=False, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)




