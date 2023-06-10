import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_file = "/Users/enzo/Documents/GitHub/dataset/MNIST/digit-recognizer/train.csv"
test_file = "/Users/enzo/Documents/GitHub/dataset/MNIST/digit-recognizer/test.csv"

train_data = pd.read_csv(train_file)
train_images = np.array(train_data.iloc[:, 1:]).reshape((-1, 28, 28))
train_labels = np.array(train_data.iloc[:, 0])

test_data = pd.read_csv(test_file)
test_images = np.array(test_data).reshape((-1, 28, 28))


# 可视化
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title(train_images[i].item())
    plt.axis("off")
    plt.imshow(train_images[i])
plt.show()




