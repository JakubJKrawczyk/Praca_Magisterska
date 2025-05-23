from torch import nn, Tensor
import torch
import math

from Custom.Layers.CNNLayer import EEGConvLayer
from Custom.Layers.FeedForwardLayer import FeedForwardNetwork
from Custom.Layers.Input import InputLayer
from Custom.Layers.MultiHeadAttention import MultiHeadAttention
from Custom.Layers.MultiLayerContainer import EncoderLayer
from Custom.Layers.PositionalEncodingLayer import PositionalEncoding


class EEG_class_model(nn.Module):
    """
    Ulepszony model klasyfikacji emocji na podstawie danych EEG
    wykorzystujący architekturę hybrydową: CNN + Transformer.

    Args:
        d_model (int): Wymiar modelu
        num_heads (int): Liczba głowic uwagi
        num_layers (int): Liczba warstw enkodera
        dropout (float): Współczynnik dropout
        num_classes (int): Liczba klas emocji do klasyfikacji
        num_bands (int): Liczba pasm częstotliwościowych (domyślnie 5)
        num_nodes (int): Liczba elektrod (domyślnie 32)
        d_ff (int, optional): Wymiar warstwy feed-forward (domyślnie 4*d_model)
        use_cnn (bool): Czy używać warstwy konwolucyjnej (domyślnie True)
    """

    def __init__(self, d_model, num_heads, num_layers=2, dropout=0.3, num_classes=7,
                 num_bands=5, num_nodes=32, d_ff=None, use_cnn=True):
        super(EEG_class_model, self).__init__()

        self.d_model = d_model
        self.num_bands = num_bands
        self.num_nodes = num_nodes
        self.use_cnn = use_cnn

        if d_ff is None:
            d_ff = d_model * 4
        self.d_ff = d_ff

        # Warstwa wejściowa
        self.input_layer = InputLayer(d_model=d_model, num_nodes=num_nodes, num_bands=num_bands)

        # Warstwa konwolucyjna (opcjonalna)
        if use_cnn:
            self.conv_layer = EEGConvLayer(
                d_model=d_model,
                num_nodes=num_nodes,
                num_bands=num_bands,
                kernel_size=3,  # Małe jądro dla lokalnych wzorców
                dropout=dropout
            )

        # Pozycyjne enkodowanie - NOWY ELEMENT
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=num_bands + num_nodes,  # Maksymalna długość to suma pasm i elektrod
            dropout=dropout
        )

        # Warstwy enkodera
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, self.d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Normalizacja końcowa
        self.layer_norm = nn.LayerNorm(d_model)

        # Współczynnik skalowania
        self.scaling_factor = nn.Parameter(torch.ones(1))

        # Warstwa uwagi między pasmami - ULEPSZONA
        self.band_attention = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)  # Softmax po wymiarze pasm
        )

        # Klasyfikator - ZWIĘKSZONY DROPOUT
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 1.5),  # Zwiększony dropout
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

        # Inicjalizacja wag - NOWY ELEMENT
        self._init_weights()

    def _init_weights(self):
        """
        Inicjalizuje wagi modelu dla lepszej zbieżności.
        """
        for name, p in self.named_parameters():
            if 'weight' in name:
                if len(p.shape) >= 2:
                    # Inicjalizacja Kaiming dla warstw liniowych i konwolucyjnych
                    nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
                else:
                    # Inicjalizacja dla biasów i innych parametrów 1D
                    nn.init.normal_(p, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def _normalize_per_sample(self, x):
        """
        Normalizuje każdą próbkę EEG niezależnie - KLUCZOWE dla transformera.

        Args:
            x (Tensor): Dane EEG [batch, bands, nodes]

        Returns:
            Tensor: Znormalizowane dane EEG
        """
        batch_size = x.shape[0]
        flat_x = x.reshape(batch_size, -1)  # [batch, bands*nodes]

        # Normalizacja Z-score dla każdej próbki
        mean = flat_x.mean(dim=1, keepdim=True)  # [batch, 1]
        std = flat_x.std(dim=1, keepdim=True) + 1e-8  # [batch, 1]
        flat_x = (flat_x - mean) / std

        # Powrót do oryginalnego kształtu
        return flat_x.reshape(x.shape)

    def forward(self, x):
        """
        Przetwarza dane EEG i klasyfikuje emocje.
        """
        # Normalizacja danych wejściowych per-sample
        x = self._normalize_per_sample(x)

        # Warstwa wejściowa
        x = self.input_layer(x)

        # Zastosowanie konwolucji, jeśli włączone
        if self.use_cnn:
            x = self.conv_layer(x)

        # Zastosowanie BEZPOŚREDNIEGO pozycyjnego enkodowania zamiast używania self.positional_encoding
        batch_size, num_bands, num_nodes, d_model = x.shape

        # Traktujemy każdą elektrodę w każdym paśmie jako oddzielny token
        x_seq = x.reshape(batch_size, num_bands * num_nodes, d_model)

        # BEZPOŚREDNIE dodanie pozycyjnego enkodowania
        seq_len = x_seq.size(1)
        device = x_seq.device

        # Generowanie pozycyjnego enkodowania na bieżąco
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, seq_len, d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Dodanie do sekwencji
        x_seq = x_seq + pe
        x_seq = nn.functional.dropout(x_seq, p=self.dropout if hasattr(self, 'dropout') else 0.1, training=self.training)

        # Powrót do formatu 4D
        x = x_seq.reshape(batch_size, num_bands, num_nodes, d_model)

        # Przetwarzanie przez warstwy enkodera
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # Normalizacja końcowa
        x = self.layer_norm(x)

        # Opcjonalne skalowanie
        x = x * self.scaling_factor

        # Agregacja informacji z elektrod poprzez uśrednianie
        # [batch, bands, nodes, d_model] -> [batch, bands, d_model]
        x = x.mean(dim=2)

        # Obliczenie wag uwagi dla każdego pasma
        band_weights = self.band_attention(x)  # [batch, bands, 1]

        # Zastosowanie wag uwagi do agregacji pasm
        x = (x * band_weights).sum(dim=1)  # [batch, d_model]

        # Klasyfikacja
        logits = self.classifier(x)

        return logits

    def get_band_attention_weights(self, x):
        """
        Zwraca wagi uwagi dla poszczególnych pasm częstotliwości.

        Args:
            x (Tensor): Tensor o kształcie [batch_size, num_bands, num_nodes]

        Returns:
            Tensor: Wagi uwagi o kształcie [batch_size, num_bands]
        """
        # Normalizacja danych wejściowych per-sample - NOWY ELEMENT
        x = self._normalize_per_sample(x)

        # Przekształcenie danych wejściowych
        x = self.input_layer(x)

        # Zastosowanie konwolucji, jeśli włączone
        if self.use_cnn:
            x = self.conv_layer(x)

        # Zastosowanie pozycyjnego enkodowania
        batch_size, num_bands, num_nodes, d_model = x.shape
        x_seq = x.reshape(batch_size, num_bands * num_nodes, d_model)
        x_seq = self.positional_encoding(x_seq)
        x = x_seq.reshape(batch_size, num_bands, num_nodes, d_model)

        # Przetwarzanie przez warstwy enkodera
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        x = self.layer_norm(x)
        x = x * self.scaling_factor

        # Agregacja po elektrodach
        x = x.mean(dim=2)  # [batch, bands, d_model]

        # Obliczenie wag uwagi
        band_weights = self.band_attention(x)  # [batch, bands, 1]

        return band_weights.squeeze(-1)  # [batch, bands]