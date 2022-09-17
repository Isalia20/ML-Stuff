import torch
from torch import nn
import torch.nn.functional as F
import math

# Simple show off on how self attention works before we get to the
# bread and butter of Self Attention. Dummy version:
# Assume we have some tensor with size x(b,t,k)
x = torch.randn(10, 3, 4)

raw_weights = torch.bmm(x, x.transpose(1, 2))
weights = F.softmax(raw_weights, dim=2)

y = torch.bmm(weights, x)


class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads
        # K and heads
        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)
        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):
        # x shape is batch, t meaning the number of words in sentence
        # and k is the dimension of vector embedding
        b, t, k = x.size()
        h = self.heads

        queries = self.toqueries(x).view(b, t, h, k)
        keys = self.tokeys(x).view(b, t, h, k)
        values = self.tovalues(x).view(b, t, h, k)

        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(k)
        # - dot has size (b * h, t, t)
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unifyheads(out)


class Transformer(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)
