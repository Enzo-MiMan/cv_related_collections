import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models


from net import vgg16, vgg16_bn
from resnet_yolo import resnet50, resnet18
from yoloLoss import yoloLoss
from dataset import yoloDataset

import numpy as np


def main():
    file_root = '/Users/enzo/Documents/GitHub/dataset/VOCdevkit/VOC2012'
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 4
    use_resnet = True
    if use_resnet:
        net = resnet50()
    else:
        net = vgg16_bn()

    # 在 backbone 后面添加 classifier
    net.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 1470),
            )

    # 给新添加的层做参数初始化
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    # 加载模型参数，继续训练
    # net.load_state_dict(torch.load('best.pth'))

    # 加载预训练好的 backbone 的参数
    print('load pre-trined model')
    if use_resnet:
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        new_state_dict = resnet.state_dict()
        dd = net.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and not k.startswith('fc'):
                dd[k] = new_state_dict[k]
        net.load_state_dict(dd)
    else:
        vgg = models.vgg16_bn(pretrained=True)
        new_state_dict = vgg.state_dict()
        dd = net.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and k.startswith('features'):
                dd[k] = new_state_dict[k]
        net.load_state_dict(dd)

    criterion = yoloLoss(7, 2, 5, 0.5)


    # different learning rate
    params=[]
    params_dict = dict(net.named_parameters())
    for key,value in params_dict.items():
        if key.startswith('features'):
            params += [{'params':[value],'lr':learning_rate*1}]
        else:
            params += [{'params':[value],'lr':learning_rate}]

    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

    train_dataset = yoloDataset(root=file_root, list_file=['my_yolo_dataset/train_label_bbox.txt'], train=True,transform = [transforms.ToTensor()] )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = yoloDataset(root=file_root, list_file=['my_yolo_dataset/val_label_bbox.txt'], train=False, transform = [transforms.ToTensor()] )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('the dataset has %d images' % (len(train_dataset)))
    print('the batch_size is %d' % (batch_size))
    logfile = open('log.txt', 'w')

    num_iter = 0
    best_test_loss = np.inf

    for epoch in range(num_epochs):
        net.train()
        if epoch == 1:
            learning_rate = 0.0005
        if epoch == 2:
            learning_rate = 0.00075
        if epoch == 3:
            learning_rate = 0.001
        if epoch == 30:
            learning_rate=0.0001
        if epoch == 40:
            learning_rate=0.00001
        # optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))

        total_loss = 0.

        for i,(images,target) in enumerate(train_loader):
            images,target = images.to(device), target.to(device)

            pred = net(images)
            loss = criterion(pred,target)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 5 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                %(epoch+1, num_epochs, i+1, len(train_loader), loss.data[0], total_loss / (i+1)))
                num_iter += 1

        #validation
        validation_loss = 0.0
        net.eval()
        for i,(images,target) in enumerate(test_loader):
            images,target = images.to(device), target.to(device)
            pred = net(images)
            loss = criterion(pred, target)
            validation_loss += loss.item()

        validation_loss /= len(test_loader)

        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(net.state_dict(),'best.pth')
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
        logfile.flush()
        torch.save(net.state_dict(),'yolo.pth')
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device : ', device)

    main()
