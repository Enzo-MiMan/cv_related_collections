import torch


def lr_lambda(x):
    return x * 2

net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1))
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

for _ in range(10):
    print(optimizer.param_groups[0]['lr'])
    optimizer.step()
    lr_scheduler.step()
    break




