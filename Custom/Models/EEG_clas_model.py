from torch import nn

from Custom.Layers.FeedForwardLayer import FeedForwardNetwork
from Custom.Layers.Input import InputLayer
from Custom.Layers.MultiLayerContainer import MultiSequenceContainer
from Custom.Layers.MultiHeadAttention import MultiHeadAttention

class EEG_class_model(nn.Module):
    def __init__(self, d_model, heads, d_ff, d_dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.input = InputLayer(expected_vector_size=self.d_model)
        self.MultiLayerContainer = MultiSequenceContainer(sequences=[
            MultiHeadAttention(num_heads=heads, d_model=self.d_model),
            FeedForwardNetwork(d_model=self.d_model, d_ff=d_ff, dropout=d_dropout),
        ])
        self.MultiLayerContainer2 = MultiSequenceContainer(sequences=[
            MultiHeadAttention(num_heads=heads, d_model=self.d_model),
            FeedForwardNetwork(d_model=self.d_model, d_ff=d_ff, dropout=d_dropout),
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.input(x)
        x = self.MultiLayerContainer.forward(x)
        x = self.MultiLayerContainer2.forward(x)
        x = self.softmax(x)
        return x