# Ulepszona warstwa FeedForwardNetwork

import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    Implementacja sieci Feed Forward Network (FFN) używanej w transformerach.

    Standardowa struktura: Linear → GELU → Dropout → Linear → Dropout → Residual → LayerNorm

    Args:
        d_model (int): Wymiar modelu
        d_ff (int, optional): Wymiar wewnętrzny sieci (zazwyczaj 4*d_model)
        dropout (float, optional): Współczynnik dropout (domyślnie 0.1)
    """

    def __init__(self, d_model, d_ff=None, dropout=0.3):
        super(FeedForwardNetwork, self).__init__()

        # Jeśli d_ff nie jest określone, używamy konwencji 4*d_model
        if d_ff is None:
            d_ff = 4 * d_model

        # Pierwsza warstwa liniowa rozszerza wymiar
        self.linear1 = nn.Linear(d_model, d_ff)

        # Aktywacja GELU (często lepsza niż ReLU dla transformerów)
        self.activation = nn.GELU()

        # Dropout po pierwszej warstwie
        self.dropout1 = nn.Dropout(dropout)

        # Druga warstwa liniowa zwęża wymiar z powrotem
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout po drugiej warstwie
        self.dropout2 = nn.Dropout(dropout)

        # Normalizacja warstwy
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Przetwarza tensor wejściowy przez sieć FFN.

        Args:
            x (Tensor): Tensor wejściowy o kształcie (..., d_model)

        Returns:
            Tensor: Tensor wyjściowy o kształcie (..., d_model)
        """
        # Zapisujemy wejście dla połączenia residualnego
        residual = x

        # Pierwsza projekcja liniowa
        x = self.linear1(x)

        # Aktywacja i dropout
        x = self.activation(x)
        x = self.dropout1(x)

        # Druga projekcja liniowa
        x = self.linear2(x)
        x = self.dropout2(x)

        # Dodanie połączenia residualnego
        x = x + residual

        # Normalizacja warstwy
        x = self.layer_norm(x)

        return x