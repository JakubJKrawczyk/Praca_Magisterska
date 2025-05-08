import torch
from torch import nn, Tensor
from Custom.Layers.ScaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Zmodyfikowana klasa MultiHeadAttention obsługująca 4D tensor [Batch, Bands, Nodes, D_model].
    Implementuje mechanizm uwagi z wieloma głowicami dla każdego pasma częstotliwości osobno.

    Args:
        num_heads (int): Liczba głowic uwagi
        d_model (int): Wymiar modelu (musi być podzielny przez num_heads)
        dropout (float): Współczynnik dropout
    """

    def __init__(self, num_heads, d_model, dropout=0.3):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout

        # Sprawdzenie, czy d_model jest podzielne przez num_heads
        assert d_model % num_heads == 0, "d_model musi być podzielne przez num_heads"
        self.d_k = d_model // num_heads  # wymiar dla każdej głowicy

        # Warstwy projekcji dla Q, K, V
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        # Warstwa wyjściowa po połączeniu
        self.output_projection = nn.Linear(d_model, d_model)

        # Uwaga i normalizacja
        self.attention = ScaledDotProductAttention(d_model=self.d_k, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def _process_band(self, x_band):
        """
        Przetwarza pojedyncze pasmo częstotliwości.

        Args:
            x_band (Tensor): Tensor o wymiarach [batch_size, num_nodes, d_model]

        Returns:
            Tensor: Przetworzone pasmo o tych samych wymiarach
        """
        batch_size, num_nodes, _ = x_band.shape

        # Projekcje Q, K, V
        Q = self.query_projection(x_band)  # [batch_size, num_nodes, d_model]
        K = self.key_projection(x_band)  # [batch_size, num_nodes, d_model]
        V = self.value_projection(x_band)  # [batch_size, num_nodes, d_model]

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

        # Obliczenie uwagi dla wszystkich głowic równolegle
        head_outputs = []
        for i in range(self.num_heads):
            head_output = self.attention(Q[:, i], K[:, i], V[:, i])  # [batch_size, num_nodes, d_k]
            head_outputs.append(head_output)

        # Łączenie wyników z różnych głowic
        multi_head_output = torch.cat(head_outputs, dim=-1)  # [batch_size, num_nodes, d_model]

        # Projekcja końcowa
        output = self.output_projection(multi_head_output)

        # Residual connection i normalizacja
        output = self.layer_norm(output + x_band)
        output = self.dropout_layer(output)

        return output

    def forward(self, x):
        """
        Przetwarzanie 4D tensora [batch, bands, nodes, d_model] poprzez
        oddzielne zastosowanie mechanizmu uwagi dla każdego pasma.

        Args:
            x (Tensor): Tensor o wymiarach [batch_size, num_bands, num_nodes, d_model]

        Returns:
            Tensor: Przetworzone dane o tych samych wymiarach
        """
        # x ma wymiary [batch_size, num_bands, num_nodes, d_model]
        batch_size, num_bands, num_nodes, d_model = x.shape

        # Sprawdzenie zgodności wymiarów
        assert d_model == self.d_model, f"Niezgodność wymiarów: wejście ma d_model={d_model}, warstwa ma d_model={self.d_model}"

        # Przygotowanie tensora wyjściowego
        output = torch.zeros_like(x)

        # Przetwarzanie każdego pasma osobno
        for band_idx in range(num_bands):
            # Wyodrębnienie danych dla pojedynczego pasma
            band_data = x[:, band_idx]  # [batch_size, num_nodes, d_model]

            # Przetworzenie pasma z użyciem mechanizmu uwagi
            processed_band = self._process_band(band_data)

            # Zapisanie wyników
            output[:, band_idx] = processed_band

        return output