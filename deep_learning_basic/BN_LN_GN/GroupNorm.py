import torch
import torch.nn as nn
import numpy as np

feature_array = np.array([[[[1, 0],  [0, 2]],
                           [[3, 4],  [1, 2]],
                           [[-2, 9], [7, 5]],
                           [[2, 3],  [4, 2]],
                           [[-2, 9], [7, 5]],
                           [[2, 3],  [4, 2]]],

                          [[[1, 2],  [-1, 0]],
                            [[1, 2], [3, 5]],
                            [[4, 7], [-6, 4]],
                            [[1, 4], [1, 5]],
                            [[4, 7], [-6, 4]],
                            [[1, 4], [1, 5]]]], dtype=np.float32)

feature_tensor = torch.tensor(feature_array.copy(), dtype=torch.float32)
gn_out = nn.GroupNorm(num_groups=3, num_channels=6)(feature_tensor)
print(gn_out)

feature_array = feature_array.reshape((2, 3, 2, 2, 2)).reshape((6, 2, 2, 2))

for i in range(feature_array.shape[0]):
    channel = feature_array[i, :, :, :]
    mean = feature_array[i, :, :, :].mean()
    var = feature_array[i, :, :, :].var()
    print(mean)
    print(var)

    feature_array[i, :, :, :] = (feature_array[i, :, :, :] - mean) / np.sqrt(var + 1e-5)
feature_array = feature_array.reshape((2, 3, 2, 2, 2)).reshape((2, 6, 2, 2))
print(feature_array)


