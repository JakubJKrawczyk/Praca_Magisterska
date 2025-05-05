# Poprawiona klasa ScaledDotProductAttention

import math
import torch
from torch import nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Implementacja mechanizmu Scaled Dot-Product Attention.

    Formuła: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        d_model (int): Wymiar modelu dla skalowania (lub d_k dla głowicy)
        dropout (float): Współczynnik dropout (domyślnie 0.1)
    """

    def __init__(self, d_model, dropout=0.3):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        """
        Oblicza mechanizm uwagi dla podanych tensorów Q, K, V.

        Args:
            q (Tensor): Tensor zapytań o kształcie (..., seq_len_q, d_model)
            k (Tensor): Tensor kluczy o kształcie (..., seq_len_k, d_model)
            v (Tensor): Tensor wartości o kształcie (..., seq_len_k, d_model)

        Returns:
            Tensor: Tensor wyniku o kształcie (..., seq_len_q, d_model)
        """
        # Sprawdzamy, czy wymiar ostatni jest zgodny
        assert q.size(-1) == k.size(-1), "Ostatni wymiar zapytań i kluczy musi być taki sam"

        # Obliczenie iloczynu skalarnego zapytań i kluczy
        # Jeśli q i k mają kształt (batch, seq_len, d_model), to wynik będzie miał kształt (batch, seq_len_q, seq_len_k)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))

        # Skalowanie wyników
        attention_scores = attention_scores / self.scale

        # Zastosowanie softmax, aby uzyskać wagi uwagi
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Zastosowanie dropout
        attention_weights = self.dropout(attention_weights)

        # Obliczenie ważonej sumy wartości
        output = torch.matmul(attention_weights, v)

        return output