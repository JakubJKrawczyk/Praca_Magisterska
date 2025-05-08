import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGConvLayer(nn.Module):
    """
    Warstwa konwolucyjna specjalnie zaprojektowana do przetwarzania danych EEG.
    Stosuje konwolucje na wymiarze elektrod (nodes) dla każdego pasma częstotliwości.

    Args:
        d_model (int): Wymiar modelu (ostatni wymiar tensora)
        num_nodes (int): Liczba elektrod (domyślnie 32)
        num_bands (int): Liczba pasm częstotliwości (domyślnie 5)
        kernel_size (int): Rozmiar kernela konwolucji (domyślnie 3)
        dropout (float): Współczynnik dropout (domyślnie 0.3)
    """

    def __init__(self, d_model, num_nodes=32, num_bands=5, kernel_size=3, dropout=0.3):
        super(EEGConvLayer, self).__init__()

        self.d_model = d_model
        self.num_nodes = num_nodes
        self.num_bands = num_bands

        # Padding dla zachowania wymiaru przestrzennego
        self.padding = kernel_size // 2

        # Konwolucje dla każdego pasma - działają na elektrodach
        self.spatial_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_model,  # Kanały wejściowe to wymiar modelu
                out_channels=d_model,  # Zachowujemy ten sam wymiar
                kernel_size=kernel_size,  # Rozmiar okna konwolucji
                padding=self.padding,  # Padding dla zachowania wymiaru
                groups=d_model // 4  # Grupy dla redukcji parametrów i lepszego wychwytywania wzorców
            ) for _ in range(num_bands)
        ])

        # Konwolucje między pasmami częstotliwości (opcjonalne)
        self.cross_band_conv = nn.Conv1d(
            in_channels=num_bands,
            out_channels=num_bands,
            kernel_size=3,
            padding=1,
            groups=1  # Pełna konwolucja między pasmami
        )

        # Normalizacja, aktywacja i dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Projekcja dla połączenia residualnego
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor o kształcie [batch_size, num_bands, num_nodes, d_model]

        Returns:
            Tensor: Tensor o tym samym kształcie z wychwyconym przestrzennym kontekstem
        """
        batch_size, num_bands, num_nodes, d_model = x.shape

        # Zapisanie wejścia dla połączenia residualnego
        residual = x

        # Przygotowanie tensora wyjściowego
        conv_output = torch.zeros_like(x)

        # Konwolucje dla każdego pasma osobno
        for band_idx in range(num_bands):
            # Pobierz dane dla tego pasma [batch, nodes, d_model]
            band_data = x[:, band_idx]

            # Zamień wymiary dla konwolucji 1D: [batch, d_model, nodes]
            band_data = band_data.transpose(1, 2)

            # Zastosuj konwolucję przestrzenną
            conv_result = self.spatial_convs[band_idx](band_data)

            # Przywróć wymiary: [batch, nodes, d_model]
            conv_result = conv_result.transpose(1, 2)

            # Zapisz wynik
            conv_output[:, band_idx] = conv_result

        # Opcjonalnie: Konwolucje między pasmami
        # Zmień kształt na [batch*nodes, bands, d_model]
        cross_band_data = conv_output.transpose(1, 3).reshape(batch_size * num_nodes, num_bands, d_model)

        # Zastosuj konwolucję między pasmami na każdym wymiarze d_model osobno
        cross_band_results = []
        for d_idx in range(d_model):
            # Pobierz "plasterek" o kształcie [batch*nodes, bands, 1]
            d_slice = cross_band_data[:, :, d_idx:d_idx + 1]
            # Zastosuj konwolucję: [batch*nodes, bands, 1]
            d_result = self.cross_band_conv(d_slice)
            cross_band_results.append(d_result)

        # Połącz wyniki z powrotem
        cross_band_output = torch.cat(cross_band_results, dim=2)  # [batch*nodes, bands, d_model]

        # Przywróć oryginalny kształt [batch, bands, nodes, d_model]
        conv_output = cross_band_output.reshape(batch_size, d_model, num_bands, num_nodes).transpose(1, 3)

        # Projekcja na oryginalną przestrzeń
        projected = self.projection(conv_output.reshape(-1, d_model)).reshape(batch_size, num_bands, num_nodes, d_model)

        # Dodanie połączenia residualnego
        output = residual + projected

        # Normalizacja, aktywacja i dropout
        output = self.layer_norm(output)
        output = self.activation(output)
        output = self.dropout(output)

        return output