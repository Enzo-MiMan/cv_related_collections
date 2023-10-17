import torch
import torch.optim.lr_scheduler

def lr_lambda(x):
    return 0.1 ** (x // 2)

net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1))
optimizer = torch.optim.SGD(net.parameters(), lr=1.0)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=2)

for epoch in range(5):
    optimizer.step()
    lr_scheduler.step()
    print('Epoch {}, LR = {}'.format(epoch, optimizer.param_groups[0]['lr']))
