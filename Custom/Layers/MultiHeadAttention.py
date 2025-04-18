import torch
from torch import nn, Tensor
from Custom.Layers.ScaledDotProductAttention import ScaledDotProductAttention
from config import device, num_heads


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout
        self.attention = ScaledDotProductAttention(d_model=self.d_model, dropout=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model, device=device)
    def forward(self, X: Tensor):
        # tensor wejściowy jest rozmiaru (32, d_model)

        tensors_to_concat = []

        for head in range(self.num_heads):

            # Wartości Q, K, V dla kolejnej glowy

            query_projection = nn.Linear(self.d_model, self.d_model, device=device)
            key_projection = nn.Linear(self.d_model, self.d_model, device=device)
            value_projection = nn.Linear(self.d_model, self.d_model, device=device)

            Q = query_projection(X)
            K = key_projection(X)
            V = value_projection(X)

            # Zastosowanie ScaledDotProductAttention
            attention = self.attention(Q, K, V)
            tensors_to_concat.append(torch.tensor(attention, dtype=torch.float32))


        concated = torch.concat(tensors_to_concat, dim=1)
        cat_fc = nn.Linear(tensors_to_concat[0].shape[1] * self.num_heads, self.d_model, device=device)

        concated = cat_fc(concated)
        concated = self.layer_norm(concated + X)

        return concated