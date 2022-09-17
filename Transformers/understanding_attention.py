import torch
import torch.nn.functional as F

# Assume we have some tensor with size x(b,t,k)
x = torch.randn(10, 3, 4)

raw_weights = torch.bmm(x, x.transpose(1, 2))
weights = F.softmax(raw_weights, dim=2)

y = torch.bmm(weights, x)
