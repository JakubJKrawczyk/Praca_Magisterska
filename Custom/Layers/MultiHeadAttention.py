import torch
from torch import nn, Tensor
from Custom.Layers.ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = 32*512
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention(d_model=self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
    def forward(self, X: Tensor):
        if X.shape != (128, 32, 512):
            raise Exception("Invalid shape")
        print("MultiHeadAttention: forward...")
        X.reshape(128, self.d_model)
        print("MultiHeadAttention: data reshaped")

        tensors_to_concat = []
        print("MultiHeadAttention: starting heads loop")

        for head in range(self.num_heads):
            # Wartości Q, K, V dla kolejnej glowy

            query_projection = nn.Linear(self.d_model, self.d_model)
            key_projection = nn.Linear(self.d_model, self.d_model)
            value_projection = nn.Linear(self.d_model, self.d_model)

            Q = query_projection(X)
            K = key_projection(X)
            V = value_projection(X)

            # Zastosowanie ScaledDotProductAttention
            attention = self.attention(Q, K, V)
            tensors_to_concat.append(attention)

        print(tensors_to_concat[0].shape)
        print("MultiHeadAttention: tensors to concat collected")

        concated = torch.concat(tensors_to_concat, dim=0)
        cat_fc = nn.Linear(tensors_to_concat[0].shape[-1], self.d_model)

        concated = cat_fc(concated)
        print("MultiHeadAttention: concatenated")
        concated = self.layer_norm(concated + X)
        print("MultiHeadAttention: forward done")

        return concated



