import torch
from torch import nn

from Custom.Layers.FeedForwardLayer import FeedForwardNetwork
from Custom.Layers.MultiHeadAttention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """
    Pojedyncza warstwa enkodera zawierająca MultiHeadAttention i FeedForwardNetwork.

    Args:
        d_model (int): Wymiar modelu
        num_heads (int): Liczba głowic uwagi
        d_ff (int, optional): Wymiar wewnętrzny sieci FFN
        dropout (float, optional): Współczynnik dropout
    """
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # MultiHeadAttention
        self.self_attention = MultiHeadAttention(num_heads, d_model, dropout)

        # FeedForwardNetwork
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, x):
        """
        Przetwarza tensor wejściowy przez warstwę enkodera.

        Args:
            x (torch.Tensor): Tensor wejściowy o kształcie (batch_size, num_nodes, d_model)

        Returns:
            torch.Tensor: Tensor wyjściowy o kształcie (batch_size, num_nodes, d_model)
        """
        # Przetwarzanie przez mechanizm uwagi
        x = self.self_attention(x)

        # Przetwarzanie przez sieć feed-forward
        x = self.feed_forward(x)

        return x

class TransformerEncoder(nn.Module):
    """
    Pełny enkoder transformera składający się z kilku warstw.

    Args:
        d_model (int): Wymiar modelu
        num_heads (int): Liczba głowic uwagi
        num_layers (int): Liczba warstw enkodera
        d_ff (int, optional): Wymiar wewnętrzny sieci FFN
        dropout (float, optional): Współczynnik dropout
    """
    def __init__(self, d_model, num_heads, num_layers, d_ff=None, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # Tworzenie listy warstw enkodera
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Końcowa normalizacja warstwy
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Przetwarza tensor wejściowy przez cały enkoder.

        Args:
            x (torch.Tensor): Tensor wejściowy o kształcie (batch_size, num_nodes, d_model)

        Returns:
            torch.Tensor: Tensor wyjściowy o kształcie (batch_size, num_nodes, d_model)
        """
        # Przetwarzanie przez każdą warstwę enkodera
        for layer in self.layers:
            x = layer(x)

        # Końcowa normalizacja
        x = self.norm(x)

        return x