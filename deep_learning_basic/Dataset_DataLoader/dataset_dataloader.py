import torch
from torch.utils import data


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.arange(0, 20)

    def __getitem__(self, index):
        x = self.data[index]
        y = x * 2
        return y

    def __len__(self):
        return len(self.data)


# 定义DataLoader
dataset = MyDataset()
print(len(dataset))    # 20
print(dataset[3])  # tensor(6)

dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=4)
print(len(dataloader))   # 5

for x in dataloader:
    print(x)

