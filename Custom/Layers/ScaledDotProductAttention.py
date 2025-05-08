import math
import torch
from torch import nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Implementacja mechanizmu Scaled Dot-Product Attention zgodna z architekturą przetwarzania EEG.

    Formuła: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Ta implementacja obsługuje różne formaty wejściowe i jest zoptymalizowana do pracy
    z danymi EEG w strukturze elektrod.

    Args:
        d_model (int): Wymiar modelu dla skalowania (d_k dla pojedynczej głowicy)
        dropout (float): Współczynnik dropout (domyślnie 0.3)
    """

    def __init__(self, d_model, dropout=0.3):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Oblicza mechanizm uwagi dla podanych tensorów Q, K, V.

        W kontekście danych EEG, q, k i v reprezentują cechy elektrod, gdzie:
        - każda elektroda (node) może zwracać uwagę na inne elektrody
        - wagi uwagi odzwierciedlają, jak istotna jest aktywność jednej elektrody
          dla interpretacji innej

        Args:
            q (Tensor): Tensor zapytań o kształcie [batch_size, num_nodes, d_k]
                reprezentujący cechy elektrod dla funkcji zapytania
            k (Tensor): Tensor kluczy o kształcie [batch_size, num_nodes, d_k]
                reprezentujący cechy elektrod dla funkcji klucza
            v (Tensor): Tensor wartości o kształcie [batch_size, num_nodes, d_k]
                reprezentujący cechy elektrod dla funkcji wartości
            mask (Tensor, optional): Opcjonalna maska dla mechanizmu uwagi
                (np. do maskowania pewnych połączeń między elektrodami)

        Returns:
            Tensor: Przetworzone dane o kształcie [batch_size, num_nodes, d_k]
                reprezentujące cechy elektrod po zastosowaniu mechanizmu uwagi
        """
        # Sprawdzamy, czy wymiar ostatni jest zgodny
        assert q.size(-1) == k.size(-1), "Ostatni wymiar zapytań i kluczy musi być taki sam"

        # Obliczenie iloczynu skalarnego zapytań i kluczy
        # Jeśli q ma kształt [batch, num_nodes_q, d_k] i k ma kształt [batch, num_nodes_k, d_k]
        # to wynik będzie miał kształt [batch, num_nodes_q, num_nodes_k]
        attention_scores = torch.matmul(q, k.transpose(-2, -1))

        # Skalowanie wyników dla stabilności
        attention_scores = attention_scores / self.scale

        # Zastosowanie maski jeśli podana (opcjonalne)
        if mask is not None:
            # Ustawiamy bardzo małą wartość dla pozycji maskowanych
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Zastosowanie softmax, aby uzyskać wagi uwagi
        # Normalizacja po wymiarze elektrod (każdy wiersz sumuje się do 1)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Zastosowanie dropout do wag (regulacja i zapobieganie przeuczeniu)
        attention_weights = self.dropout(attention_weights)

        # Obliczenie ważonej sumy wartości
        # Każda elektroda otrzymuje kombinację cech innych elektrod
        # ważoną przez wartości uwagi
        output = torch.matmul(attention_weights, v)

        return output