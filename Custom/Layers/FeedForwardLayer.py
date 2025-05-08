import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    Implementacja sieci Feed Forward Network (FFN) dostosowana do przetwarzania danych EEG.
    Obsługuje format 4D [Batch, Bands, Nodes, D_model] i przetwarza każde pasmo częstotliwości osobno.

    Struktura: Linear → GELU → Dropout → Linear → Dropout → Residual → LayerNorm

    Args:
        d_model (int): Wymiar modelu (ostatni wymiar tensora wejściowego)
        d_ff (int, optional): Wymiar wewnętrzny sieci (zazwyczaj 4*d_model)
        dropout (float, optional): Współczynnik dropout (domyślnie 0.3)
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

    def _process_band(self, x_band):
        """
        Przetwarza pojedyncze pasmo częstotliwości.

        Args:
            x_band (Tensor): Tensor pasma o kształcie [batch_size, num_nodes, d_model]

        Returns:
            Tensor: Przetworzone pasmo o tym samym kształcie
        """
        # Zapisujemy wejście dla połączenia residualnego
        residual = x_band

        # Pierwsza projekcja liniowa
        x = self.linear1(x_band)

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

    def forward(self, x):
        """
        Przetwarza 4D tensor [batch, bands, nodes, d_model] przez sieć FFN,
        osobno dla każdego pasma częstotliwości.

        Args:
            x (Tensor): Tensor wejściowy o kształcie [batch_size, num_bands, num_nodes, d_model]
                reprezentujący cechy elektrod dla różnych pasm częstotliwości

        Returns:
            Tensor: Tensor wyjściowy o takim samym kształcie
        """
        # x ma wymiary [batch_size, num_bands, num_nodes, d_model]
        batch_size, num_bands, num_nodes, d_model = x.shape

        # Przygotowanie tensora wyjściowego
        output = torch.zeros_like(x)

        # Przetwarzanie każdego pasma osobno
        for band_idx in range(num_bands):
            # Wyodrębnienie danych dla pojedynczego pasma
            band_data = x[:, band_idx]  # [batch_size, num_nodes, d_model]

            # Przetworzenie pasma
            processed_band = self._process_band(band_data)

            # Zapisanie wyników
            output[:, band_idx] = processed_band

        return output