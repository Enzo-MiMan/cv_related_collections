import torch.nn as nn
import torch
import matplotlib.pyplot as plt


class Self_Attention(nn.Module):
    def __init__(self, dim):
        super(Self_Attention, self).__init__()
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, -1).permute(2, 0, 1, 3)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        return x

att = Self_Attention(dim=2)
x = torch.rand((1, 4, 2))
output = att(x)
















# class MultiHead_Attention(nn.Module):
#     def __init__(self, dim, num_heads):
#
#         super(MultiHead_Attention, self).__init__()
#         self.num_heads = num_heads   # 2
#         head_dim = dim // num_heads   # 2
#         self.scale = head_dim ** -0.5   # 1
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
# att = MultiHead_Attention(dim=768, num_heads=12)
# x = torch.rand((1, 197, 768))
# output = att(x)


