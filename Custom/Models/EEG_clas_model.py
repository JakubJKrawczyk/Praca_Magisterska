from torch import nn

from Custom.Layers.FeedForwardLayer import FeedForwardNetwork
from Custom.Layers.Input import InputLayer
from Custom.Layers.MultiLayerContainer import MultiSequenceContainer
from Custom.Layers.MultiHeadAttention import MultiHeadAttention

class EEG_class_model(nn.Module):
    def __init__(self, d_model = 32, *args, **kwargs):
        super().__init__(*args, **kwargs, heads=8, d_model=32)
        self.model = None
        self.d_model = d_model
        self.input = InputLayer(batch=128, expected_vector_size=self.d_model)
        self.MultiLayerContainer = MultiSequenceContainer(sequences=[
            MultiHeadAttention(num_heads=4),
            FeedForwardNetwork(d_model=self.d_model),
        ])
        self.MultiLayerContainer2 = MultiSequenceContainer(sequences=[
            MultiHeadAttention(num_heads=4),
            FeedForwardNetwork(d_model=self.d_model),
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.input(x)
        x = self.MultiLayerContainer.forward(x)
        x = self.MultiLayerContainer2.forward(x)
        x = self.softmax(x)
        return x