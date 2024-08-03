import torch
from torch import nn
import torch.nn.functional as F


def scaled_dot_product(q, k, v):
    d = q.shape[-1]
    weight = q @ k.transpose(-2, -1) / torch.sqrt(d)
    attention = F.softmax(weight, dim=-1)
    return attention @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model

        self.W = nn.Linear(d_model, 3 * d_model, bias=False)  # qkv matrix
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len)))

    def forward(self, x):
        n, t, c = x.shape  # likely, [n, t, c]
        h = self.W(x)

        # split into heads
        h = h.view(
            n, t, self.num_heads, 3 * c // self.num_heads
        )  # [n, t, c] -> [n, t, h, 3*dims]
        h = h.permute(0, 2, 1, 3)

        # split into q, k, v
        q, k, v = torch.split(h, 3, dim=-1)
