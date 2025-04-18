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
        x = self.input(x)
        x = self.MultiLayerContainer.forward(x)
        x = self.MultiLayerContainer2.forward(x)

        # Użycie zdefiniowanych parametrów
        x = x * self.scaling_factor

        # Przykład użycia parametrów do klasyfikacji
        # Zakładając, że x ma wymiar [batch_size, d_model]
        # możemy użyć parametrów do klasyfikacji
        logits = torch.matmul(x, self.class_weights) + self.bias

        logits = logits.mean(dim=0, keepdim=True)

        x = self.softmax(logits)
        return x