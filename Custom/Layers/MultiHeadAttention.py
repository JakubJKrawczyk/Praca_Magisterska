# Zmodyfikowana klasa MultiHeadAttention

import torch
from torch import nn, Tensor
from Custom.Layers.ScaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.3):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout

        # Sprawdzenie, czy d_model jest podzielne przez num_heads
        assert d_model % num_heads == 0, "d_model musi być podzielne przez num_heads"

        self.d_k = d_model // num_heads  # wymiar dla każdej głowicy

        # Warstwy projekcji dla Q, K, V (tworzone RAZ w __init__, nie w forward)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        # Warstwa wyjściowa po połączeniu
        self.output_projection = nn.Linear(d_model, d_model)

        # Uwaga i normalizacja
        self.attention = ScaledDotProductAttention(d_model=self.d_k, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # x ma wymiary [batch_size, num_nodes, d_model]
        batch_size, num_nodes, _ = x.shape

        # Projekcje Q, K, V
        Q = self.query_projection(x)  # [batch_size, num_nodes, d_model]
        K = self.key_projection(x)  # [batch_size, num_nodes, d_model]
        V = self.value_projection(x)  # [batch_size, num_nodes, d_model]

        # Zmiana kształtu dla multi-head attention
        # Dzielimy ostatni wymiar na num_heads i d_k
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.d_k)
        K = K.view(batch_size, num_nodes, self.num_heads, self.d_k)
        V = V.view(batch_size, num_nodes, self.num_heads, self.d_k)

        # Transpozycja dla obliczenia uwagi na każdej głowicy
        # [batch_size, num_heads, num_nodes, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Obliczenie uwagi - w pętli dla każdej głowicy
        head_outputs = []
        for i in range(self.num_heads):
            head_output = self.attention(Q[:, i], K[:, i], V[:, i])
            head_outputs.append(head_output)

        # Łączenie wyników z różnych głowic
        multi_head_output = torch.cat(head_outputs, dim=-1)  # [batch_size, num_nodes, d_model]

        # Projekcja końcowa
        output = self.output_projection(multi_head_output)

        # Residual connection i normalizacja
        output = self.layer_norm(output + x)
        output = self.dropout_layer(output)

        return output