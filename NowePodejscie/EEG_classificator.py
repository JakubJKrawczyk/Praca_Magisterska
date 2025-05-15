import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class TemporalSpatialBlock(nn.Module):
    def __init__(self, in_channels, temporal_filters, spatial_filters, num_probes=62):
        super(TemporalSpatialBlock, self).__init__()

        # Temporal convolution
        self.temporal_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=temporal_filters,
            kernel_size=(9, 1),  # Większy kernel dla lepszego kontekstu czasowego
            stride=1,
            padding=(4, 0),
            bias=False
        )

        # Spatial convolution
        self.spatial_conv = nn.Conv2d(
            in_channels=temporal_filters,
            out_channels=spatial_filters,
            kernel_size=(1, 9),  # Większy kernel dla lepszych wzorców przestrzennych
            stride=1,
            padding=(0, 4),
            groups=1,  # Grupowanie dla lepszego przetwarzania przestrzennego
            bias=False
        )

        # Normalizacje
        self.layer_norm1 = nn.GroupNorm(8, temporal_filters)  # Zamiast BatchNorm
        self.layer_norm2 = nn.GroupNorm(8, spatial_filters)

        # Squeeze-and-Excitation
        self.se = SEBlock(spatial_filters)

        # Residual connection
        self.residual = nn.Conv2d(in_channels, spatial_filters, kernel_size=1,
                                  stride=1, padding=0, bias=False) if in_channels != spatial_filters else nn.Identity()

        # Aktywacje
        self.elu = nn.ELU(inplace=True)

        # Dropout
        self.dropout1 = nn.Dropout2d(0.2)  # Spatial dropout
        self.dropout2 = nn.Dropout2d(0.2)

    def forward(self, x):
        residual = self.residual(x)

        # Temporal branch
        x = self.temporal_conv(x)
        x = self.layer_norm1(x)
        x = self.elu(x)
        x = self.dropout1(x)

        # Spatial branch
        x = self.spatial_conv(x)
        x = self.layer_norm2(x)
        x = self.elu(x)
        x = self.dropout2(x)

        # Apply SE
        x = self.se(x)

        # Residual connection
        x = x + residual
        x = self.elu(x)

        return x


class BidirectionalLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super(BidirectionalLSTMWithAttention, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # Bidirectional dla lepszego kontekstu
            dropout=dropout if num_layers > 1 else 0
        )

        # Self-attention dla sekwencji LSTM
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 dla dwukierunkowego LSTM
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size*2]

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)

        # Residual connection + norm
        out = self.norm(lstm_out + attn_out)

        return out


class WindowBasedEEGClassifier(nn.Module):
    def __init__(self, window_size=128, window_stride=16, num_frequency_bands=5, num_probes=62,
                 num_classes=2, temporal_filters=64, spatial_filters=128,
                 transformer_dim=256, nhead=8, num_layers=4, dropout=0.3):
        super(WindowBasedEEGClassifier, self).__init__()

        self.window_size = window_size
        self.window_stride = window_stride
        self.num_frequency_bands = num_frequency_bands
        self.num_probes = num_probes

        # LSTM dla każdego pasma częstotliwości
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=num_probes,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=dropout
            )
            for _ in range(num_frequency_bands)
        ])

        # Temporal-Spatial Block dla przetwarzania przestrzenno-czasowego
        # Przyjmuje dane w formacie [batch, channels=frequency_bands, width=time, height=features]
        self.feature_extractor = nn.Sequential(
            TemporalSpatialBlock(num_frequency_bands, temporal_filters, temporal_filters, num_probes),
            TemporalSpatialBlock(temporal_filters, temporal_filters, spatial_filters, num_probes)
        )

        # BiLSTM with attention do zaawansowanego przetwarzania sekwencyjnego
        self.bilstm_attn = BidirectionalLSTMWithAttention(
            input_size=spatial_filters * num_probes,
            hidden_size=transformer_dim // 2,  # Dzielenie przez 2, ponieważ BiLSTM podwaja wymiar
            num_layers=2,
            dropout=dropout
        )

        # Positional encoding dla transformera
        self.pos_encoder = PositionalEncoding(transformer_dim, max_len=window_size)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global context aggregation
        self.global_attention_pool = nn.Sequential(
            nn.Linear(transformer_dim, 1),
            nn.Softmax(dim=1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.LayerNorm(transformer_dim // 2),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        Przetwarza dane EEG z wykorzystaniem okna czasowego.

        Args:
            x: Tensor o kształcie [batch, bands, window_size, probes]
                gdzie window_size to długość okna czasowego

        Returns:
            logits: Predykcje dla klas
        """
        # Wejście: [batch, bands, window_size, probes]
        batch_size = x.size(0)

        # Przetwarzanie każdego pasma częstotliwości przez LSTM
        processed_bands = []
        for band_idx in range(self.num_frequency_bands):
            # Pobierz dane dla jednego pasma
            band_data = x[:, band_idx, :, :]  # [batch, window_size, probes]

            # Zastosuj LSTM
            band_lstm_out, _ = self.lstm_layers[band_idx](band_data)
            # band_lstm_out: [batch, window_size, hidden_size=64]

            processed_bands.append(band_lstm_out)

        # Łączymy przetworzone pasma w jeden tensor 4D
        # [batch, bands, window_size, hidden_size]
        x_lstm = torch.stack(processed_bands, dim=1)

        # Ekstrakcja cech przestrzenno-czasowych za pomocą CNN
        # [batch, bands, window_size, hidden_size] -> [batch, filters, window_size, features]
        x = self.feature_extractor(x_lstm)

        # Zmiana kształtu dla przetwarzania sekwencyjnego
        # [batch, filters, window_size, features] -> [batch, window_size, features*filters]
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, self.window_size, -1)

        # BiLSTM z attention - zaawansowane przetwarzanie sekwencyjne
        x = self.bilstm_attn(x)

        # Positional encoding i transformer
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x)

        # Global attention pooling - agregacja kontekstu
        attn_weights = self.global_attention_pool(transformer_out)
        context_vector = torch.sum(attn_weights * transformer_out, dim=1)

        # Klasyfikacja
        output = self.classifier(context_vector)

        return output

    def predict_with_sliding_window(self, eeg_data, overlap=0.75):
        """
        Wykonuje predykcję używając techniki przesuwnego okna.

        Args:
            eeg_data: Tensor o kształcie [bands, time_samples, probes]
                      Cały sygnał EEG
            overlap: Stopień nakładania się okien (domyślnie 0.75)

        Returns:
            predictions: Średnie predykcje ze wszystkich okien
            all_logits: Predykcje dla każdego okna
        """
        # Parametry okna
        window_size = self.window_size
        stride = int(window_size * (1 - overlap))  # Przesunięcie okna

        # Wymiary sygnału
        bands, total_time, probes = eeg_data.shape

        # Liczba okien
        n_windows = max(1, (total_time - window_size) // stride + 1)

        # Predykcje dla każdego okna
        all_logits = []

        # Przetwarzanie każdego okna
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = min(start_idx + window_size, total_time)

            # Jeśli okno jest krótsze niż window_size, uzupełnij zerami
            if end_idx - start_idx < window_size:
                padding_size = window_size - (end_idx - start_idx)
                padding = torch.zeros((bands, padding_size, probes), device=eeg_data.device)
                window_data = torch.cat([eeg_data[:, start_idx:end_idx, :], padding], dim=1)
            else:
                # Wytnij okno z sygnału
                window_data = eeg_data[:, start_idx:end_idx, :]

            # Dodaj wymiar batcha
            window_data = window_data.unsqueeze(0)  # [1, bands, window_size, probes]

            # Wykonaj predykcję
            with torch.no_grad():
                logits = self(window_data)
                all_logits.append(logits)

        # Złącz wyniki ze wszystkich okien
        all_logits = torch.cat(all_logits, dim=0)

        # Uśrednij predykcje
        avg_prediction = torch.mean(all_logits, dim=0, keepdim=True)

        return avg_prediction, all_logits