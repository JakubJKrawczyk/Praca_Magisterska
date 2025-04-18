from torch import nn, Tensor
import torch
from Custom.Layers.FeedForwardLayer import FeedForwardNetwork
from Custom.Layers.Input import InputLayer
from Custom.Layers.MultiLayerContainer import MultiSequenceContainer
from Custom.Layers.MultiHeadAttention import MultiHeadAttention

class EEG_class_model(nn.Module):
    def __init__(self, d_model, heads, d_dropout, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scaling_factor = nn.Parameter(torch.ones(1))
        self.class_weights = nn.Parameter(torch.randn(d_model, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

        self.d_model = d_model
        self.input = InputLayer(expected_vector_size=self.d_model)
        self.MultiLayerContainer = MultiSequenceContainer(layers=[
            MultiHeadAttention(num_heads=heads, d_model=self.d_model, dropout=d_dropout),
            FeedForwardNetwork(d_model=self.d_model, dropout=d_dropout),
        ])
        self.MultiLayerContainer2 = MultiSequenceContainer(layers=[
            MultiHeadAttention(num_heads=heads, d_model=self.d_model, dropout=d_dropout),
            FeedForwardNetwork(d_model=self.d_model, dropout=d_dropout),
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor):
        # Jeśli x ma kształt [batch_size, seq_len, features]
        # Przetwarzamy każdy przykład w batchu osobno
        if x.dim() == 3:
            batch_size = x.size(0)
            outputs = []
            
            for i in range(batch_size):
                # Przetwarzanie pojedynczego przykładu
                single_x = x[i]  # [seq_len, features]
                single_x = self.input(single_x)
                single_x = self.MultiLayerContainer.forward(single_x)
                single_x = self.MultiLayerContainer2.forward(single_x)
                
                # Użycie zdefiniowanych parametrów
                single_x = single_x * self.scaling_factor
                
                # Uśredniamy reprezentacje sekwencji
                single_x_mean = single_x.mean(dim=0, keepdim=True)  # [1, d_model]
                
                # Klasyfikacja
                logits = torch.matmul(single_x_mean, self.class_weights) + self.bias  # [1, num_classes]
                outputs.append(logits)
            
            # Łączymy wyniki dla całego batcha
            return torch.cat(outputs, dim=0)  # [batch_size, num_classes]
        else:
            # Przetwarzanie pojedynczego przykładu (gdy x ma kształt [seq_len, features])
            x = self.input(x)
            x = self.MultiLayerContainer.forward(x)
            x = self.MultiLayerContainer2.forward(x)
            
            # Użycie zdefiniowanych parametrów
            x = x * self.scaling_factor

        # Uśredniamy reprezentacje sekwencji
        # x ma kształt [seq_len, d_model]
        x_mean = x.mean(dim=0, keepdim=True)  # [1, d_model]
        
        # Klasyfikacja na podstawie uśrednionej reprezentacji
        logits = torch.matmul(x_mean, self.class_weights) + self.bias  # [1, num_classes]

        x = self.softmax(logits)
        return x
