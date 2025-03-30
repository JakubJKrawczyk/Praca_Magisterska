import jax.numpy as jnp
from pyarrow import Tensor
from torch import nn, torch

def expand_value_sinusoidal(input_tensor, size=512) -> Tensor:
    """
    Metoda 1: Kodowanie sinusoidalne - podobne do kodowania pozycyjnego w transformerach.
    Przekształca pojedynczą wartość w wektor za pomocą funkcji sinusoidalnych o różnej częstotliwości.
    """
    if input_tensor is None or input_tensor.shape != (128,32,1):
        raise Exception("Input tensor must be of shape (128,32,1)")

    result = torch.zeros((input_tensor.size(0), input_tensor.size(1), size), dtype=jnp.float32)
    freqs = 10 ** (jnp.arange(0, size // 2) / (size // 2 - 1) * 4.0)  # Częstotliwości od 1 do 10000
    for sample in range(input_tensor.size(0)):
        for node in range(input_tensor.size(1)):
            for expected_value in range(size // 2):
                result[sample][node][2 * expected_value] = torch.tensor(jnp.sin(input_tensor[sample][node] * freqs[expected_value]))
                result[sample][node][2 * expected_value + 1] = torch.tensor(jnp.cos(input_tensor * freqs[expected_value]))

    return result


class InputLayer(nn.Module):
    def __init__(self, batch, expected_vector_size):
        super(InputLayer, self).__init__()
        self.size = expected_vector_size
        self.inputBatch = batch
        self.expand_values = expand_value_sinusoidal
    def forward(self, x):
        result = self.expand_values(x, self.size)
        print("InputLayer forward")
        if result.shape == (128,32,self.size):
            return result
        else:
            yield Exception("Invalid shape")
