from torch import nn, Tensor
import torch
import math

from Custom.Layers.CNNLayer import EEGConvLayer
from Custom.Layers.FeedForwardLayer import FeedForwardNetwork
from Custom.Layers.Input import InputLayer
from Custom.Layers.MultiHeadAttention import MultiHeadAttention
from Custom.Layers.MultiLayerContainer import EncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):  # Zwiększ max_len do dużej wartości
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Wyświetl informacje debugowania przy inicjalizacji
        print(f"Initializing PositionalEncoding with max_len={max_len}, d_model={d_model}")

        # Tworzenie stałej tablicy pozycyjnego enkodowania
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # Rejestracja jako bufor (nie parametr)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Wyświetl informacje o rozmiarach podczas przetwarzania
        print(f"PositionalEncoding forward: x.shape={x.shape}, pe.shape={self.pe.shape}")

        # Sprawdź, czy wymiary są zgodne przed dodaniem
        if x.size(1) > self.pe.shape[1]:
            # Dynamicznie rozszerz pe do wymaganego rozmiaru
            device = x.device
            required_len = x.size(1)
            new_pe = torch.zeros(1, required_len, self.pe.shape[2], device=device)

            # Wypełnij nowy tensor używając wartości z istniejącego pe
            position = torch.arange(0, required_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.pe.shape[2], 2, device=device).float() * (-math.log(10000.0) / self.pe.shape[2]))

            new_pe[0, :, 0::2] = torch.sin(position * div_term)
            new_pe[0, :, 1::2] = torch.cos(position * div_term)

            # Użyj nowego pe zamiast starego
            print(f"Dynamically expanded pe to shape {new_pe.shape}")
            x = x + new_pe[:, :x.size(1), :]
        else:
            # Standardowe dodanie
            x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)


class EEG_class_model(nn.Module):
    """
    Ulepszony model klasyfikacji emocji na podstawie danych EEG 
    wykorzystujący architekturę hybrydową: CNN + Transformer.
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
                kernel_size=3,
                dropout=dropout
            )

        # Pozycyjne enkodowanie - POPRAWIONE
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=num_bands * num_nodes,  # POPRAWIONE: używamy mnożenia zamiast dodawania
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
            nn.Softmax(dim=1)
        )

        # Klasyfikator
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 1.5),
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

        # Inicjalizacja wag
        self._init_weights()

    def _init_weights(self):
        """
        Inicjalizuje wagi modelu dla lepszej zbieżności.
        """
        for name, p in self.named_parameters():
            if 'weight' in name:
                if len(p.shape) >= 2:
                    nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.normal_(p, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def _normalize_per_sample(self, x):
        """
        Normalizuje każdą próbkę EEG niezależnie.
        """
        batch_size = x.shape[0]
        flat_x = x.reshape(batch_size, -1)
        mean = flat_x.mean(dim=1, keepdim=True)
        std = flat_x.std(dim=1, keepdim=True) + 1e-8
        flat_x = (flat_x - mean) / std
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

        # Zastosowanie pozycyjnego enkodowania
        batch_size, num_bands, num_nodes, d_model = x.shape

        # Traktujemy każdą elektrodę w każdym paśmie jako oddzielny token
        x_seq = x.reshape(batch_size, num_bands * num_nodes, d_model)

        # Dodajemy pozycyjne enkodowanie
        x_seq = self.positional_encoding(x_seq)

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
        x = x.mean(dim=2)

        # Obliczenie wag uwagi dla każdego pasma
        band_weights = self.band_attention(x)

        # Zastosowanie wag uwagi do agregacji pasm
        x = (x * band_weights).sum(dim=1)

        # Klasyfikacja
        logits = self.classifier(x)

        return logits

    def get_band_attention_weights(self, x):
        """
        Zwraca wagi uwagi dla poszczególnych pasm częstotliwości.
        """
        # Normalizacja danych wejściowych per-sample
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
        x = x.mean(dim=2)

        # Obliczenie wag uwagi
        band_weights = self.band_attention(x)

        return band_weights.squeeze(-1)