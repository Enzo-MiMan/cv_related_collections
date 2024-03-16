'''
代码来自 : https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
'''

import torch
import math
import matplotlib.pyplot as plt


def positional_encoding(d_model, length):
    """
    :param d_model: dimension of the token
    :param length: (maximum) token number
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


pe = positional_encoding(128, 10)
plt.plot(range(10), pe[:, 0])
plt.show()


