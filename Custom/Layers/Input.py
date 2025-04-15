import jax.numpy as jnp
from pyarrow import Tensor
from torch import nn, torch

def expand_value_sinusoidal(input_tensor, vector_size = 512) -> Tensor:
    """
    Metoda 1: Kodowanie sinusoidalne - podobne do kodowania pozycyjnego w transformerach.
    Przekształca pojedynczą wartość w wektor za pomocą funkcji sinusoidalnych o różnej częstotliwości.
    """
    if input_tensor is None:
        raise Exception("Input tensor is NONE")

    result = torch.zeros((input_tensor.size(0), vector_size), dtype=jnp.float32)
    freqs = 10 ** (jnp.arange(0, vector_size // 2) / (vector_size // 2 - 1) * 4.0)  # Częstotliwości od 1 do 10000
    for sample in range(input_tensor.size(0)):
            for expected_value in range(vector_size // 2):
                result[sample][2 * expected_value] = torch.tensor(jnp.sin(input_tensor[sample] * freqs[expected_value]))
                result[sample][2 * expected_value + 1] = torch.tensor(jnp.cos(input_tensor * freqs[expected_value]))

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
        if result.ndim[1].size == self.size:
            return result
        else:
            yield Exception("Invalid shape")
