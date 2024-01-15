from PIL import  Image
import torch
import os
import torch.utils.data as data


if __name__ == '__main__':

    batch_size = 16

    class train_dataset(data.Dataset):
        def __init__(self, path, labels, trans):
            pass

        def __getitem__(self, item):
            ...
            return image, target

        def __len__(self):
            pass

        @staticmethod
        def collate_fn(batch):
            images, targets = tuple(zip(*batch))
            images = torch.stack(images, dim=0)
            return images, targets


    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    for images, targets in train_loader:
        pass

