import torch.nn as nn
import torch


class AdditiveAttention(nn.Module):
    def __init__(self, query_size, key_size, num_hiddens, dropout):
        super(AdditiveAttention, self).__init__()
        self.w_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.w_q(queries), self.w_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        return features


att = AdditiveAttention(3, 6, 5, False)
queries = torch.rand((1, 4, 3))
keys = torch.rand((1, 4, 6))
values = torch.rand((1, 4, 5))
out = att(queries, keys, values, None)







