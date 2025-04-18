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
        # Przetwarzamy dane wejściowe przez warstwę wejściową
        # x ma kształt [batch_size, features]
        x = self.input(x)  # Teraz x ma kształt [batch_size, d_model]
        
        # Dla każdego przykładu w batchu
        batch_size = x.size(0)
        outputs = []
        
        for i in range(batch_size):
            # Pobierz pojedynczy wektor cech
            single_x = x[i].unsqueeze(0)  # [1, d_model]
            
            # Przetwarzanie przez warstwy transformera
            # Uwaga: MultiLayerContainer oczekuje wejścia [seq_len, d_model]
            # Tutaj seq_len = 1, więc możemy użyć bezpośrednio
            single_x = self.MultiLayerContainer.forward(single_x)
            single_x = self.MultiLayerContainer2.forward(single_x)
            
            # Użycie zdefiniowanych parametrów
            single_x = single_x * self.scaling_factor
            
            # Klasyfikacja (single_x już ma kształt [1, d_model])
            logits = torch.matmul(single_x, self.class_weights) + self.bias  # [1, num_classes]
            outputs.append(logits)
        
        # Łączymy wyniki dla całego batcha
        return torch.cat(outputs, dim=0)  # [batch_size, num_classes]

        # Ta część nie będzie używana, ponieważ zawsze przetwarzamy dane jako batch
        # Ale zostawiamy ją dla kompatybilności
        x = self.softmax(logits)
        return x
