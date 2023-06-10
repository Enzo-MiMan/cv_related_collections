import torchvision.datasets
import matplotlib.pyplot as plt


data_path = "/Users/enzo/Documents/GitHub/dataset"

train_dataset = torchvision.datasets.MNIST(data_path, train=True, download=True)
test_dataset = torchvision.datasets.MNIST(data_path, train=False, download=False)

train_images = train_dataset.data
train_labels = train_dataset.targets
test_images = train_dataset.data
test_labels = train_dataset.targets

# 可视化
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title(train_labels[i].item())
    plt.axis("off")
    plt.imshow(train_images[i])
plt.show()
