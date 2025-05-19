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
        # Store the input size for debugging
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Self-attention for LSTM sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Print dimensions for debugging
        print(f"BiLSTM input shape: {x.shape}, expected features: {self.input_size}")

        # LSTM processing - FIXED SYNTAX
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)

        # Residual connection + norm
        out = self.norm(lstm_out + attn_out)

        return out


class AdaptiveEEGClassifier(nn.Module):
    def __init__(self, window_size=5, num_frequency_bands=5, num_probes=32,
                 num_classes=7, dropout=0.3):
        super(AdaptiveEEGClassifier, self).__init__()

        self.window_size = window_size
        self.num_frequency_bands = num_frequency_bands
        self.num_probes = num_probes

        # Basic LSTM for each frequency band
        hidden_size = 64
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=num_probes,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0
            )
            for _ in range(num_frequency_bands)
        ])

        # Simple feature extractor (more robust than complex CNNs)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_frequency_bands, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Feature dimension calculation (will be determined during first forward pass)
        self.feature_dim = None

        # Final classifier (will be initialized during first forward pass)
        self.classifier = None
        self.num_classes = num_classes
        self.dropout = dropout

    def _initialize_classifier(self, feature_dim):
        """Initializes the classifier with the correct input dimension"""
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes)
        ).to(next(self.parameters()).device)  # Ensure it's on the same device
        print(f"Initialized classifier with input dimension: {feature_dim}")

    def forward(self, x):
        # Input debugging
        # print(f"Input tensor shape: {x.shape}")

        # Handle different input formats
        if x.dim() == 5:  # [batch, windows, bands, time, probes]
            batch_size, num_windows = x.size(0), x.size(1)
            x = x.reshape(-1, x.size(2), x.size(3), x.size(4))
        else:  # [batch, bands, time, probes]
            batch_size, num_windows = x.size(0), 1

        # Process each band with LSTM
        processed_bands = []
        for band_idx in range(self.num_frequency_bands):
            band_data = x[:, band_idx, :, :]  # [batch*windows, time, probes]

            try:
                # Process with LSTM
                band_lstm_out, _ = self.lstm_layers[band_idx](band_data)

                # Use either the entire sequence or just the last output
                if band_data.size(1) > 1:
                    # Average across time dimension for robustness
                    band_feat = torch.mean(band_lstm_out, dim=1)
                else:
                    band_feat = band_lstm_out.squeeze(1)

                processed_bands.append(band_feat)

            except Exception as e:
                print(f"Error processing band {band_idx}: {e}")
                print(f"Band data shape: {band_data.shape}")
                print(f"LSTM expected input size: {self.lstm_layers[band_idx].input_size}")
                raise

        # Combine band features
        combined_features = torch.cat(processed_bands, dim=1)

        # Initialize classifier if needed
        if self.classifier is None:
            self._initialize_classifier(combined_features.size(1))

        # Final classification
        output = self.classifier(combined_features)

        # Reshape output if needed for multiple windows
        if num_windows > 1:
            output = output.view(batch_size, num_windows, -1)
            output = output.mean(dim=1)  # Average across windows

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