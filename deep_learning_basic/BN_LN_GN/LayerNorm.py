import torch
import torch.nn as nn
import numpy as np

feature_array = np.array([[[[1, 0],  [0, 2]],
                           [[3, 4],  [1, 2]],
                           [[2, 3],  [4, 2]]],

                          [[[1, 2],  [-1, 0]],
                            [[1, 2], [3, 5]],
                            [[1, 4], [1, 5]]]], dtype=np.float32)


feature_array = feature_array.reshape((2, 3, -1)).transpose(0, 2, 1)
feature_tensor = torch.tensor(feature_array.copy(), dtype=torch.float32)

ln_out = nn.LayerNorm(normalized_shape=3)(feature_tensor)
print(ln_out)

b, token_num, dim = feature_array.shape
feature_array = feature_array.reshape((-1, dim))
for i in range(b*token_num):
    mean = feature_array[i, :].mean()
    var = feature_array[i, :].var()
    print(mean)
    print(var)

    feature_array[i, :] = (feature_array[i, :] - mean) / np.sqrt(var + 1e-5)
print(feature_array.reshape(b, token_num, dim))
