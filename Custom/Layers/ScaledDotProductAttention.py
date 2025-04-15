import math
from torch import nn, torch
from torch.nn import functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1, *args, **kwargs):
        super(ScaledDotProductAttention, self).__init__(*args, **kwargs)
        self.sqrt_d_model = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        first = torch.matmul(q, k.transpose(1, 2))
        second = first / self.sqrt_d_model
        third = F.softmax(second, dim=-1)
        fourth = self.dropout(third)
        return torch.matmul(fourth, v)
