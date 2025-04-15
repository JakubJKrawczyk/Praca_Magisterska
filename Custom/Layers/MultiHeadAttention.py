import torch
from torch import nn, Tensor
from Custom.Layers.ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention(d_model=self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
    def forward(self, X: Tensor):
        # tensor wejściowy jest rozmiaru (32, d_model)
        print("MultiHeadAttention: forward...")

        tensors_to_concat = []
        print("MultiHeadAttention: starting heads loop")

        for head in range(self.num_heads):
            nodeset = []
            for node in range(X.shape[0]):

                # Wartości Q, K, V dla kolejnej glowy

                query_projection = nn.Linear(self.d_model, self.d_model)
                key_projection = nn.Linear(self.d_model, self.d_model)
                value_projection = nn.Linear(self.d_model, self.d_model)

                Q = query_projection(X[node])
                K = key_projection(X[node])
                V = value_projection(X[node])

                # Zastosowanie ScaledDotProductAttention
                attention = self.attention(Q, K, V)
                nodeset.append(attention)
            tensors_to_concat.append(torch.tensor(nodeset, dtype=torch.float32))

        print(tensors_to_concat[0].shape)
        print("MultiHeadAttention: tensors to concat collected")

        concated = torch.concat(tensors_to_concat, dim=0)
        cat_fc = nn.Linear(tensors_to_concat[0].shape[-1], self.d_model)

        concated = cat_fc(concated)
        print("MultiHeadAttention: concatenated")
        concated = self.layer_norm(concated + X)
        print("MultiHeadAttention: forward done")

        return concated



