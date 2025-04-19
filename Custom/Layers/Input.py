import torch
from torch import nn
import math

class InputLayer(nn.Module):
    """
    Warstwa wejściowa przekształcająca odczyty z elektrod EEG na reprezentację o wyższym wymiarze,
    zachowując strukturę elektrod.

    Args:
        expected_vector_size (int): Oczekiwany wymiar wektora cech dla każdej elektrody
        num_nodes (int, optional): Liczba elektrod/węzłów (domyślnie 32)
        dropout_rate (float, optional): Współczynnik dropout (domyślnie 0.1)
    """
    def __init__(self, expected_vector_size, num_nodes=32, dropout_rate=0.1):
        super(InputLayer, self).__init__()
        self.size = expected_vector_size
        self.num_nodes = num_nodes

        # Tworzymy warstwę liniową dla każdej elektrody - ta sama warstwa będzie używana
        # dla wszystkich elektrod (współdzielone wagi)
        self.node_projection = nn.Linear(1, expected_vector_size)

        # Inicjalizacja wag dla lepszej stabilności uczenia
        nn.init.xavier_uniform_(self.node_projection.weight)
        nn.init.zeros_(self.node_projection.bias)

        # Normalizacja warstwy i dropout
        self.norm = nn.LayerNorm(expected_vector_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Skalowanie dla stabilniejszego uczenia
        self.register_buffer('scale', torch.tensor(math.sqrt(expected_vector_size)))

    # Add this to your data processing pipeline
    def normalize_eeg_data(self, values_tensor):
        # Normalize each sample individually
        mean = values_tensor.mean(dim=1, keepdim=True)
        std = values_tensor.std(dim=1, keepdim=True)
        return (values_tensor - mean) / (std + 1e-8)
    def expand_values(self, input_tensor):
        """
        Rozszerza każdą wartość z elektrod EEG do wektora cech o wymiarze expected_vector_size.

        Args:
            input_tensor (torch.Tensor): Tensor o kształcie (batch_size, num_nodes)

        Returns:
            torch.Tensor: Tensor o kształcie (batch_size, num_nodes, expected_vector_size)
        """
        if input_tensor is None:
            raise ValueError("Input tensor is None")

        batch_size, num_nodes = input_tensor.shape

        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}")

        # Zmiana kształtu tensora wejściowego (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        input_tensor = input_tensor.unsqueeze(-1)
        # self.normalize_eeg_data(input_tensor.values())

        # Pusta tablica na wyniki
        output = torch.zeros(batch_size, num_nodes, self.size, device=input_tensor.device)

        # Przekształcenie każdej wartości na wektor cech
        for i in range(num_nodes):
            node_values = input_tensor[:, i, :]  # (batch_size, 1)
            # Używamy wspólnej warstwy liniowej dla wszystkich elektrod
            node_embedding = self.node_projection(node_values)  # (batch_size, expected_vector_size)
            output[:, i, :] = node_embedding

        # Skalowanie dla stabilniejszego uczenia
        output = output * self.scale

        return output

    def forward(self, x):
        """
        Przetwarza tensor wejściowy, przekształcając każdą wartość z elektrody na wektor cech.

        Args:
            x (torch.Tensor): Tensor o kształcie (batch_size, num_nodes) zawierający odczyty z elektrod

        Returns:
            torch.Tensor: Tensor o kształcie (batch_size, num_nodes, expected_vector_size)
        """
        # Handle 1D input (single sample case)
        if x.dim() == 1:
            # Add batch dimension
            x = x.unsqueeze(0)

        # Check dimensions
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[1]}")

        # Rozszerzenie wartości do wektorów cech
        output = self.expand_values(x)

        # Normalizacja i dropout
        output = self.norm(output)
        output = self.dropout(output)

        return output