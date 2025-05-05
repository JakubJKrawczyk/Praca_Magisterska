# Ulepszony model EEG_class_model

from torch import nn, Tensor
import torch
from Custom.Layers.FeedForwardLayer import FeedForwardNetwork
from Custom.Layers.Input import InputLayer
from Custom.Layers.MultiHeadAttention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """
    Pojedyncza warstwa enkodera zawierająca MultiHeadAttention i FeedForwardNetwork.
    """

    def __init__(self, d_model, num_heads, dropout=0.3):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout)
        self.feed_forward = FeedForwardNetwork(d_model=d_model, dropout=dropout)

    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class EEG_class_model(nn.Module):
    """
    Model klasyfikacji emocji na podstawie danych EEG wykorzystujący architekturę transformera.

    Args:
        d_model (int): Wymiar modelu
        num_heads (int): Liczba głowic uwagi
        num_layers (int): Liczba warstw enkodera
        dropout (float): Współczynnik dropout
        num_classes (int): Liczba klas emocji do klasyfikacji
    """

    def __init__(self, d_model, num_heads, num_layers=2, dropout=0.3, num_classes=7):
        super(EEG_class_model, self).__init__()

        # Warstwa wejściowa
        self.input_layer = InputLayer(expected_vector_size=d_model)

        # Warstwy enkodera
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Normalizacja końcowa
        self.layer_norm = nn.LayerNorm(d_model)

        # Współczynnik skalowania (opcjonalny)
        self.scaling_factor = nn.Parameter(torch.ones(1))

        # Klasyfikator
        self.classifier = nn.Sequential(
            # Uwaga: Potrzebujemy spłaszczyć dane z elektrod
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        Przetwarza dane EEG i klasyfikuje emocje.

        Args:
            x (Tensor): Tensor o kształcie (batch_size, 32) zawierający odczyty z elektrod EEG

        Returns:
            Tensor: Tensor prawdopodobieństw klas o kształcie (batch_size, num_classes)
        """
        # Normalizacja danych wejściowych
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        x = (x - mean) / std

        # Warstwa wejściowa - przekształca dane na (batch_size, 32, d_model)
        x = self.input_layer(x)

        # Przetwarzanie przez warstwy enkodera
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # Normalizacja końcowa
        x = self.layer_norm(x)

        # Opcjonalne skalowanie
        x = x * self.scaling_factor

        # Uśrednianie po elektrodach - Global Average Pooling
        # Teraz x ma kształt (batch_size, d_model)
        x = x.mean(dim=1)

        # Klasyfikacja
        logits = self.classifier(x)

        # Zwracamy logity - nie stosujemy softmax, ponieważ funkcja straty CrossEntropyLoss już go zawiera
        return logits