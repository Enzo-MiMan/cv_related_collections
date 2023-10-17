import torch
#
# device = "cpu"

# def criterion(predict, target):
#     pass
# def data_loader(predict, target):
#     pass
#
#
#
#
# def lr_lambda(x):
#     return 0.95 ** x
#
# optimizer = torch.optim.SGD(model.parameters, lr=0.001, momentum=0.9)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
#
# for epoch in range(30):
#     model.train()
#
#     for image, target in data_loader:
#         image, target = image.to(device), target.to(device)
#
#         predict = model(image)
#         loss = criterion(predict, target)
#
#         optimizer.zero_grad()
#
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#
#         ...


def lr_lambda(x):
    return x * 2

net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1))
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

for _ in range(10):
    optimizer.step()
    lr_scheduler.step()
    print(optimizer.param_groups[0]['lr'])

