import torch.nn as nn


class Alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Flatten(), nn.Linear(256 * 5 * 5, 4096), nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 4096), nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 10))

    def forward(self, X):
        return self.net(X)


net = Alexnet()
print(net)

