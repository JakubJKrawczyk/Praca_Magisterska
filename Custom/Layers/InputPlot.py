import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import numpy as np

class InputLayer(nn.Module):
    """
    Warstwa wejściowa przekształcająca odczyty z elektrod EEG na reprezentację o wyższym wymiarze,
    zachowując strukturę elektrod.

    Args:
        expected_vector_size (int): Oczekiwany wymiar wektora cech dla każdej elektrody
        num_nodes (int, optional): Liczba elektrod/węzłów (domyślnie 32)
        dropout_rate (float, optional): Współczynnik dropout (domyślnie 0.1)
    """
    def __init__(self, expected_vector_size, num_nodes=32, dropout_rate=0.1):
        super(InputLayer, self).__init__()
        self.size = expected_vector_size
        self.num_nodes = num_nodes

        # Tworzymy warstwę liniową dla każdej elektrody - ta sama warstwa będzie używana
        # dla wszystkich elektrod (współdzielone wagi)
        self.node_projection = nn.Linear(1, expected_vector_size)

        # Inicjalizacja wag dla lepszej stabilności uczenia
        nn.init.xavier_uniform_(self.node_projection.weight)
        nn.init.zeros_(self.node_projection.bias)

        # Normalizacja warstwy i dropout
        self.norm = nn.LayerNorm(expected_vector_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Skalowanie dla stabilniejszego uczenia
        self.register_buffer('scale', torch.tensor(math.sqrt(expected_vector_size)))

        # Zmienna do śledzenia statystyk
        self.track_stats = False
        self.input_stats = []
        self.output_stats = []

    # Add this to your data processing pipeline
    def normalize_eeg_data(self, values_tensor):
        # Normalize each sample individually
        mean = values_tensor.mean(dim=1, keepdim=True)
        std = values_tensor.std(dim=1, keepdim=True)
        return (values_tensor - mean) / (std + 1e-8)

    def expand_values(self, input_tensor):
        """
        Rozszerza każdą wartość z elektrod EEG do wektora cech o wymiarze expected_vector_size.

        Args:
            input_tensor (torch.Tensor): Tensor o kształcie (batch_size, num_nodes)

        Returns:
            torch.Tensor: Tensor o kształcie (batch_size, num_nodes, expected_vector_size)
        """
        if input_tensor is None:
            raise ValueError("Input tensor is None")

        batch_size, num_nodes = input_tensor.shape

        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}")

        # Jeśli śledzenie statystyk jest włączone, zapisz statystyki wejścia
        if self.track_stats:
            # Konwertuj na CPU i numpy dla statystyk
            input_np = input_tensor.detach().cpu().numpy()
            self.input_stats.append({
                'mean': np.mean(input_np),
                'std': np.std(input_np),
                'min': np.min(input_np),
                'max': np.max(input_np),
                'histogram': np.histogram(input_np.flatten(), bins=50)
            })

        # Zmiana kształtu tensora wejściowego (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        input_tensor = input_tensor.unsqueeze(-1)

        # Pusta tablica na wyniki
        output = torch.zeros(batch_size, num_nodes, self.size, device=input_tensor.device)

        # Przekształcenie każdej wartości na wektor cech
        for i in range(num_nodes):
            node_values = input_tensor[:, i, :]  # (batch_size, 1)
            # Używamy wspólnej warstwy liniowej dla wszystkich elektrod
            node_embedding = self.node_projection(node_values)  # (batch_size, expected_vector_size)
            output[:, i, :] = node_embedding

        # Skalowanie dla stabilniejszego uczenia
        output = output * self.scale

        # Jeśli śledzenie statystyk jest włączone, zapisz statystyki wyjścia
        if self.track_stats:
            # Konwertuj na CPU i numpy dla statystyk
            output_np = output.detach().cpu().numpy()
            self.output_stats.append({
                'mean': np.mean(output_np),
                'std': np.std(output_np),
                'min': np.min(output_np),
                'max': np.max(output_np),
                'histogram': np.histogram(output_np.flatten(), bins=50)
            })

        return output

    def forward(self, x):
        """
        Przetwarza tensor wejściowy, przekształcając każdą wartość z elektrody na wektor cech.

        Args:
            x (torch.Tensor): Tensor o kształcie (batch_size, num_nodes) zawierający odczyty z elektrod

        Returns:
            torch.Tensor: Tensor o kształcie (batch_size, num_nodes, expected_vector_size)
        """
        # Handle 1D input (single sample case)
        if x.dim() == 1:
            # Add batch dimension
            x = x.unsqueeze(0)

        # Check dimensions
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[1]}")

        # Rozszerzenie wartości do wektorów cech
        output = self.expand_values(x)

        # Normalizacja i dropout
        output = self.norm(output)
        output = self.dropout(output)

        return output

    def enable_stats_tracking(self):
        """Włącza śledzenie statystyk dla danych wejściowych i wyjściowych"""
        self.track_stats = True
        self.input_stats = []
        self.output_stats = []

    def disable_stats_tracking(self):
        """Wyłącza śledzenie statystyk"""
        self.track_stats = False

    def plot_data_distribution(self, figsize=(15, 10)):
        """
        Generuje wykres rozkładu danych przed i po przejściu przez warstwę liniową.

        Args:
            figsize (tuple): Rozmiar wykresu (szerokość, wysokość)

        Returns:
            matplotlib.figure.Figure: Figura z wykresami
        """
        if not self.track_stats or len(self.input_stats) == 0:
            raise ValueError("No statistics available. Enable stats tracking with enable_stats_tracking() first.")

        # Utwórz figurę
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Przygotuj dane do histogramów
        combined_input = np.concatenate([stats['histogram'][0] for stats in self.input_stats])
        input_bins = self.input_stats[0]['histogram'][1]

        combined_output = np.concatenate([stats['histogram'][0] for stats in self.output_stats])
        output_bins = self.output_stats[0]['histogram'][1]

        # Średnie wartości przed warstwą liniową
        input_means = [stats['mean'] for stats in self.input_stats]
        input_stds = [stats['std'] for stats in self.input_stats]

        # Średnie wartości po warstwie liniowej (przed normalizacją)
        output_means = [stats['mean'] for stats in self.output_stats]
        output_stds = [stats['std'] for stats in self.output_stats]

        # Histogram dla danych wejściowych
        axes[0, 0].hist(np.concatenate([stats['histogram'][1][:-1] for stats in self.input_stats]),
                        weights=combined_input, bins=50, alpha=0.7)
        axes[0, 0].set_title('Rozkład danych wejściowych')
        axes[0, 0].set_xlabel('Wartość')
        axes[0, 0].set_ylabel('Częstość')

        # Histogram dla danych po warstwie liniowej
        axes[0, 1].hist(np.concatenate([stats['histogram'][1][:-1] for stats in self.output_stats]),
                        weights=combined_output, bins=50, alpha=0.7)
        axes[0, 1].set_title('Rozkład danych po warstwie liniowej')
        axes[0, 1].set_xlabel('Wartość')
        axes[0, 1].set_ylabel('Częstość')

        # Porównanie minimalnych i maksymalnych wartości
        batch_indices = range(len(self.input_stats))

        # Min/Max dla danych wejściowych
        axes[0, 2].plot(batch_indices, [stats['min'] for stats in self.input_stats], 'b-', label='Min')
        axes[0, 2].plot(batch_indices, [stats['max'] for stats in self.input_stats], 'r-', label='Max')
        axes[0, 2].set_title('Min/Max wartości wejściowych')
        axes[0, 2].set_xlabel('Indeks batcha')
        axes[0, 2].set_ylabel('Wartość')
        axes[0, 2].legend()

        # Średnie i odchylenia standardowe dla danych wejściowych
        axes[1, 0].plot(batch_indices, input_means, 'g-', label='Średnia')
        axes[1, 0].fill_between(batch_indices,
                                [m - s for m, s in zip(input_means, input_stds)],
                                [m + s for m, s in zip(input_means, input_stds)],
                                alpha=0.3, color='g')
        axes[1, 0].set_title('Średnia ± Odch. std. (wejście)')
        axes[1, 0].set_xlabel('Indeks batcha')
        axes[1, 0].set_ylabel('Wartość')
        axes[1, 0].legend()

        # Średnie i odchylenia standardowe dla danych po warstwie liniowej
        axes[1, 1].plot(batch_indices, output_means, 'g-', label='Średnia')
        axes[1, 1].fill_between(batch_indices,
                                [m - s for m, s in zip(output_means, output_stds)],
                                [m + s for m, s in zip(output_means, output_stds)],
                                alpha=0.3, color='g')
        axes[1, 1].set_title('Średnia ± Odch. std. (po warstwie liniowej)')
        axes[1, 1].set_xlabel('Indeks batcha')
        axes[1, 1].set_ylabel('Wartość')
        axes[1, 1].legend()

        # Min/Max dla danych po warstwie liniowej
        axes[1, 2].plot(batch_indices, [stats['min'] for stats in self.output_stats], 'b-', label='Min')
        axes[1, 2].plot(batch_indices, [stats['max'] for stats in self.output_stats], 'r-', label='Max')
        axes[1, 2].set_title('Min/Max wartości po warstwie liniowej')
        axes[1, 2].set_xlabel('Indeks batcha')
        axes[1, 2].set_ylabel('Wartość')
        axes[1, 2].legend()

        plt.tight_layout()
        return fig

    def save_distribution_plot(self, filename='eeg_data_distribution.png'):
        """
        Zapisuje wykres rozkładu danych do pliku.

        Args:
            filename (str): Nazwa pliku, do którego zostanie zapisany wykres
        """
        fig = self.plot_data_distribution()
        fig.savefig(filename)
        plt.close(fig)
        print(f"Wykres zapisany do {filename}")


# Przykład użycia:
if __name__ == "__main__":
    # Utwórz warstwę wejściową
    input_layer = InputLayer(expected_vector_size=64, num_nodes=32)

    # Włącz śledzenie statystyk
    input_layer.enable_stats_tracking()

    # Symuluj dane EEG dla kilku batchów
    for i in range(5):
        # Generuj losowe dane (wartości EEG zazwyczaj mieszczą się w zakresie ±100µV)
        fake_eeg_data = torch.randn(16, 32) * 50  # 16 próbek, 32 elektrody

        # Przetwórz dane przez warstwę wejściową
        output = input_layer(fake_eeg_data)

    # Wygeneruj i zapisz wykres
    input_layer.save_distribution_plot()

    # Możesz również wyświetlić wykres w trybie interaktywnym:
    # plt.show(input_layer.plot_data_distribution())