import torch
from torch import nn
import math


class InputLayer(nn.Module):
    """
    Ulepszona warstwa wejściowa przetwarzająca sygnały EEG z uwzględnieniem pasm częstotliwości.
    Przyjmuje tensor o wymiarach [Batch, Bands, Nodes] i zwraca [Batch, Bands, Nodes, D_model].

    Args:
        d_model (int): Wymiar wektora cech dla każdej elektrody i każdego pasma
        num_nodes (int): Liczba elektrod (domyślnie 32)
        num_bands (int): Liczba pasm częstotliwości (domyślnie 5 - delta, theta, alpha, beta, gamma)
        dropout_rate (float): Współczynnik dropout
    """

    def __init__(self, d_model, num_nodes=32, num_bands=5, dropout_rate=0.3):
        super(InputLayer, self).__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.num_bands = num_bands

        # Projekcja indywidualna dla każdej wartości elektrody
        self.node_embedding = nn.Linear(1, d_model // 2)

        # Projekcja dla każdego pasma - uwzględnia relacje między elektrodami
        self.band_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_nodes, (d_model // 2) * num_nodes),
                nn.GELU()
            ) for _ in range(num_bands)
        ])

        # Warstwa łącząca dwa rodzaje embeddingów
        self.combination_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )

        # Normalizacja i dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

        # Skalowanie dla stabilności uczenia
        self.register_buffer('scale', torch.tensor(math.sqrt(d_model)))

    def forward(self, x):
        """
        Przetwarza tensor wejściowy, zachowując strukturę pasm i elektrod.

        Args:
            x (torch.Tensor): Tensor o kształcie (batch_size, num_bands, num_nodes)
                zawierający odczyty z elektrod EEG

        Returns:
            torch.Tensor: Tensor o kształcie (batch_size, num_bands, num_nodes, d_model)
        """
        # Sprawdzenie wymiarów wejściowych
        if x.dim() != 3:
            raise ValueError(f"Oczekiwano tensora 3D [batch, bands, nodes], otrzymano kształt {x.shape}")

        batch_size, num_bands, num_nodes = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Oczekiwano {self.num_nodes} elektrod, otrzymano {num_nodes}")
        if num_bands != self.num_bands:
            raise ValueError(f"Oczekiwano {self.num_bands} pasm częstotliwości, otrzymano {num_bands}")

        # Przygotowanie wyjściowego tensora: [batch, bands, nodes, d_model]
        output = torch.zeros(batch_size, num_bands, num_nodes, self.d_model, device=x.device)

        # 1. Indywidualne przetwarzanie każdej wartości elektrody
        # Reshape do [batch * bands * nodes, 1]
        x_flat = x.reshape(-1, 1)

        # Projekcja do [batch * bands * nodes, d_model//2]
        node_embeddings = self.node_embedding(x_flat)

        # Reshape z powrotem do [batch, bands, nodes, d_model//2]
        node_embeddings = node_embeddings.reshape(batch_size, num_bands, num_nodes, self.d_model // 2)

        # 2. Przetwarzanie uwzględniające relacje między elektrodami w paśmie
        for i in range(num_bands):
            # Wartości dla jednego pasma: [batch_size, num_nodes]
            band_values = x[:, i, :]

            # Projekcja uwzględniająca relacje między elektrodami: [batch_size, (d_model//2) * num_nodes]
            band_context = self.band_projections[i](band_values)

            # Reshape do [batch_size, num_nodes, d_model//2]
            band_context = band_context.reshape(batch_size, num_nodes, self.d_model // 2)

            # Łączenie z indywidualnymi embeddingami
            combined = torch.cat([
                node_embeddings[:, i],  # [batch, nodes, d_model//2]
                band_context  # [batch, nodes, d_model//2]
            ], dim=-1)  # [batch, nodes, d_model]

            # Finalne przetworzenie
            final_embedding = self.combination_layer(combined)  # [batch, nodes, d_model]

            # Zapisanie do wyjściowego tensora
            output[:, i] = final_embedding

        # Skalowanie, normalizacja i dropout
        output = output * self.scale
        output = self.norm(output)
        output = self.dropout(output)

        return output