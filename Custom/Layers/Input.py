from torch import nn, torch
import math

def expand_value_sinusoidal(input_tensor: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Rozwija tensor (x, 32) do (x, 32, d_model) za pomocą sinusoidalnego kodowania.
    Dla każdej wartości wejściowej tworzy wektor sinusoidalny długości d_model.
    """
    if input_tensor is None:
        raise Exception("Input tensor is NONE")

    num_sensors = input_tensor.shape[0]  # 1, 32

    # Częstotliwości dla sinusoid
    position = input_tensor.unsqueeze(-1)  # (32, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
    ).to("cuda")  # (d_model/2)

    # Obliczamy sinusoidalne kodowanie
    sinusoid_input = position * div_term  # (x, 32, d_model/2)
    sin_embed = torch.sin(sinusoid_input)
    cos_embed = torch.cos(sinusoid_input)

    # Tworzymy pusty tensor i przeplatamy sinusoidy
    output = torch.zeros(num_sensors, d_model, dtype=torch.float32).to("cuda")
    output[..., 0::2] = sin_embed
    output[..., 1::2] = cos_embed

    return output

class InputLayer(nn.Module):
    def __init__(self, expected_vector_size):
        super(InputLayer, self).__init__()
        self.size = expected_vector_size
        self.expand_values = expand_value_sinusoidal
    def forward(self, x):
        result = self.expand_values(x, self.size)
        if result.shape[1] == self.size:
            return result
        else:
            raise Exception("Invalid shape")
