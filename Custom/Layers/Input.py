# Nowa wersja Input.py

import torch
from torch import nn
import math


class InputLayer(nn.Module):
    """
    Ulepszona warstwa wejściowa przetwarzająca sygnały EEG z uwzględnieniem pasm częstotliwości.

    Args:
        expected_vector_size (int): Oczekiwany wymiar wektora wyjściowego
        num_nodes (int): Liczba elektrod (domyślnie 32)
        num_bands (int): Liczba pasm częstotliwości (domyślnie 5 - delta, theta, alpha, beta, gamma)
        dropout_rate (float): Współczynnik dropout
    """

    def __init__(self, expected_vector_size, num_nodes=32, num_bands=5, dropout_rate=0.3):
        super(InputLayer, self).__init__()
        self.size = expected_vector_size
        self.num_nodes = num_nodes
        self.num_bands = num_bands

        # Zakładamy, że dane wejściowe mają kształt [batch_size, num_nodes]
        # i chcemy je przekształcić na [batch_size, num_nodes, expected_vector_size]

        # Warstwa liniowa dla każdego pasma częstotliwości
        band_size = expected_vector_size // num_bands
        self.band_projections = nn.ModuleList([
            nn.Linear(1, band_size) for _ in range(num_bands)
        ])

        # Warstwa do łączenia cech z różnych pasm
        self.combine_bands = nn.Linear(band_size * num_bands, expected_vector_size)

        # Normalizacja i dropout
        self.norm = nn.LayerNorm(expected_vector_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Skalowanie dla stabilności uczenia
        self.register_buffer('scale', torch.tensor(math.sqrt(expected_vector_size)))

    def forward(self, x):
        """
        Przetwarza tensor wejściowy, uwzględniając strukturę pasm częstotliwości.

        Args:
            x (torch.Tensor): Tensor o kształcie (batch_size, num_nodes) lub (batch_size, num_bands, num_nodes)
                zawierający odczyty z elektrod EEG

        Returns:
            torch.Tensor: Tensor o kształcie (batch_size, num_nodes, expected_vector_size)
        """
        # Obsługa wymiarów wejściowych
        if x.dim() == 2:
            # Jeśli wejście to (batch_size, num_nodes), przekształcamy je na
            # (batch_size, 1, num_nodes) i powielamy dla każdego pasma
            batch_size, num_nodes = x.shape
            if num_nodes != self.num_nodes:
                raise ValueError(f"Oczekiwano {self.num_nodes} elektrod, otrzymano {num_nodes}")

            # Dodanie wymiaru dla pasm częstotliwości
            x = x.unsqueeze(1).expand(-1, self.num_bands, -1)

        elif x.dim() == 3:
            # Jeśli wejście to już (batch_size, num_bands, num_nodes)
            batch_size, num_bands, num_nodes = x.shape
            if num_nodes != self.num_nodes:
                raise ValueError(f"Oczekiwano {self.num_nodes} elektrod, otrzymano {num_nodes}")
            if num_bands != self.num_bands:
                raise ValueError(f"Oczekiwano {self.num_bands} pasm częstotliwości, otrzymano {num_bands}")
        else:
            raise ValueError(f"Oczekiwano tensora 2D lub 3D, otrzymano kształt {x.shape}")

        # Przetwarzanie każdego pasma częstotliwości osobno
        band_outputs = []
        for i in range(self.num_bands):
            # Wartości dla jednego pasma: [batch_size, num_nodes]
            band_values = x[:, i, :]

            # Przekształcenie na [batch_size, num_nodes, 1]
            band_values = band_values.unsqueeze(-1)

            # Projekcja na wektor cech dla każdej elektrody
            band_features = self.band_projections[i](band_values)  # [batch_size, num_nodes, band_size]
            band_outputs.append(band_features)

        # Łączenie cech z różnych pasm dla każdej elektrody
        combined_bands = torch.cat(band_outputs, dim=-1)  # [batch_size, num_nodes, band_size * num_bands]

        # Końcowa projekcja do oczekiwanego wymiaru
        output = self.combine_bands(combined_bands)  # [batch_size, num_nodes, expected_vector_size]

        # Skalowanie, normalizacja i dropout
        output = output * self.scale
        output = self.norm(output)
        output = self.dropout(output)

        return output