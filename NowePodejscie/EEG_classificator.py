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


class ImprovedEEGClassifier(nn.Module):
    def __init__(self, input_time_samples, num_frequency_bands=5, num_probes=62,
                 num_classes=2, temporal_filters=64, spatial_filters=128,
                 transformer_dim=256, nhead=8, num_layers=4, dropout=0.3):
        super(ImprovedEEGClassifier, self).__init__()

        self.input_time_samples = input_time_samples
        self.num_frequency_bands = num_frequency_bands
        self.num_probes = num_probes

        # Temporal-Spatial Block for feature extraction
        self.feature_extractor = nn.Sequential(
            TemporalSpatialBlock(num_frequency_bands, temporal_filters, temporal_filters, num_probes),
            TemporalSpatialBlock(temporal_filters, temporal_filters, spatial_filters, num_probes)
        )

        # BiLSTM with attention
        self.bilstm_attn = BidirectionalLSTMWithAttention(
            input_size=spatial_filters * num_probes,
            hidden_size=transformer_dim // 2,  # Dzielenie przez 2, ponieważ BiLSTM podwaja wymiar
            num_layers=2,
            dropout=dropout
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(transformer_dim, max_len=input_time_samples)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation="gelu",  # GELU for transformer is often better
            batch_first=True,
            norm_first=True  # Pre-norm architecture dla stabilności
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global context aggregation
        self.global_attention_pool = nn.Sequential(
            nn.Linear(transformer_dim, 1),
            nn.Softmax(dim=1)
        )

        # Classifier with multiple layers
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.LayerNorm(transformer_dim // 2),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, num_classes)
        )

    def forward(self, x):
        # Wejście: [batch, time_samples, frequency_bands, probes]
        batch_size = x.size(0)

        # Permutacja do [batch, frequency_bands, time_samples, probes]
        x = x.permute(0, 2, 1, 3)

        # Ekstrakcja cech z CNN
        x = self.feature_extractor(x)

        # Zmiana kształtu dla modelu sekwencyjnego: [batch, time_samples, features]
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, self.input_time_samples, -1)

        # BiLSTM z attention
        x = self.bilstm_attn(x)

        # Positional encoding + transformer
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x)

        # Attention pooling dla lepszej agregacji kontekstu
        attn_weights = self.global_attention_pool(transformer_out)
        context_vector = torch.sum(attn_weights * transformer_out, dim=1)

        # Klasyfikacja
        output = self.classifier(context_vector)

        return output