import torch.nn as nn
from pyarrow import Tensor


class MultiSequenceContainer(nn.Module):
    def __init__(self, layers):
        super(MultiSequenceContainer, self).__init__()
        self.layers = layers

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dLdy):
        for layer in reversed(self.layers):
            dLdy = layer.backward(dLdy)
        return dLdy

    def step(self, learning_rate):
        for layer in self.layers:
            layer.step(learning_rate)

    def zero_grad(self, **kwargs):
        for layer in self.layers:
            layer.zero_grad()