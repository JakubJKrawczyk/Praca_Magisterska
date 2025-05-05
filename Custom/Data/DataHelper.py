from operator import contains, indexOf
import scipy.io as sio
import torch
from pandas.core.interchange.utils import dtype_to_arrow_c_fmt
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class EmotionDataset(Dataset):
    """
    Dataset przechowujący dane EEG wraz z etykietami emocji.

    Każdy element datasetu to para (emotion_id, electrode_values), gdzie:
    - emotion_id: identyfikator emocji (0-6)
    - electrode_values: tensor o wymiarze (32,) zawierający wartości z 32 elektrod
    """
    def __init__(self, emotion_data:[{int, np.ndarray}]):
        self.emotions = []
        self.values = []
        self.idx_to_emotion = {}

        # Przypisujemy indeksy do emocji
        for idx, emotion_array in enumerate(emotion_data):
            self.idx_to_emotion[idx] = mapuj_emocje(emotion_array["emotion_id"], False)

        # Wypełniamy listy danych
        for emotion_array in emotion_data:
            self.emotions.append(mapuj_emocje(emotion_array["emotion_id"], True))
            self.values.append(emotion_array["value"])

        # Wyświetlenie informacji o danych
        if len(self.values) > 0:
            print(f"EmotionDataset created with {len(self.emotions)} samples")
            print(f"Each sample contains values from 32 electrodes: {len(self.values[0])}")

    def tensorize(self):
        """
        Konwertuje dane na tensory PyTorch i zwraca TensorDataset.

        Returns:
            TensorDataset: Dataset zawierający parę tensorów (emotions, values)
                emotions: tensor o wymiarze (n_samples,) - identyfikatory emocji
                values: tensor o wymiarze (n_samples, 32) - wartości z elektrod
        """
        # Konwertujemy na tensory
        emotions_tensor = torch.tensor(self.emotions, dtype=torch.long)
        values_tensor = DataHelper.normalize_de_features(torch.tensor(self.values, dtype=torch.float))


# Sprawdzenie wymiarów
        print(f"Tensors created: emotions {emotions_tensor.shape}, values {values_tensor.shape}")

        # Tworzenie TensorDataset
        return TensorDataset(values_tensor, emotions_tensor)

    def __len__(self):
        return len(self.emotions)

    def get_emotion(self, idx):
        return self.idx_to_emotion[idx]

    def extend(self, other):
        """
        Rozszerza dataset o dane z innego EmotionDataset.

        Args:
            other (EmotionDataset): Dataset do dodania
        """
        if isinstance(other, EmotionDataset):
            self.emotions.extend(other.emotions)
            self.values.extend(other.values)
            incrementer = len(self.idx_to_emotion)
            for emotion in other.idx_to_emotion.values():
                self.idx_to_emotion[incrementer] = emotion
                incrementer += 1
        else:
            raise TypeError("Can only add another EmotionDataset instance.")


# Mapowanie ID emocji na nazwy
emotionsEnum = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Anger",
    4: "Fear",
    5: "Surprise",
    6: "Disgust"
}

positiveNegative_enum = {
    0: "Negative",
    1: "Positive"
}

def mapuj_emocje(liczba : int, return_id : bool):
    if liczba in [1, 5, 0]:
        if return_id:
            return 1
        else:
            return positiveNegative_enum[1]
    elif liczba in [2, 3, 4, 6]:
        if return_id:
            return 0
        else:
            return positiveNegative_enum[0]

# Mapowanie ID wideo na emocje
VideoIdToEmotionMap = {
    # Sesja 1 (1-20)
    1: 1,  # Happy
    2: 0,  # Neutral
    3: 6,  # Disgust
    4: 2,  # Sad
    5: 3,  # Anger
    6: 3,  # Anger
    7: 2,  # Sad
    8: 6,  # Disgust
    9: 0,  # Neutral
    10: 1,  # Happy
    11: 1,  # Happy
    12: 0,  # Neutral
    13: 6,  # Disgust
    14: 2,  # Sad
    15: 3,  # Anger
    16: 3,  # Anger
    17: 2,  # Sad
    18: 6,  # Disgust
    19: 0,  # Neutral
    20: 1,  # Happy
    # Sesja 2 (21-40)
    21: 3,  # Anger
    22: 2,  # Sad
    23: 4,  # Fear
    24: 0,  # Neutral
    25: 5,  # Surprise
    26: 5,  # Surprise
    27: 0,  # Neutral
    28: 4,  # Fear
    29: 2,  # Sad
    30: 3,  # Anger
    31: 3,  # Anger
    32: 2,  # Sad
    33: 4,  # Fear
    34: 0,  # Neutral
    35: 5,  # Surprise
    36: 5,  # Surprise
    37: 0,  # Neutral
    38: 4,  # Fear
    39: 2,  # Sad
    40: 3,  # Anger
    # Sesja 3 (41-60)
    41: 1,  # Happy
    42: 5,  # Surprise
    43: 6,  # Disgust
    44: 4,  # Fear
    45: 3,  # Anger
    46: 3,  # Anger
    47: 4,  # Fear
    48: 6,  # Disgust
    49: 5,  # Surprise
    50: 1,  # Happy
    51: 1,  # Happy
    52: 5,  # Surprise
    53: 6,  # Disgust
    54: 4,  # Fear
    55: 3,  # Anger
    56: 3,  # Anger
    57: 4,  # Fear
    58: 6,  # Disgust
    59: 5,  # Surprise
    60: 1,  # Happy
    # Sesja 4 (61-80)
    61: 6,  # Disgust
    62: 2,  # Sad
    63: 4,  # Fear
    64: 5,  # Surprise
    65: 1,  # Happy
    66: 1,  # Happy
    67: 5,  # Surprise
    68: 4,  # Fear
    69: 2,  # Sad
    70: 6,  # Disgust
    71: 6,  # Disgust
    72: 2,  # Sad
    73: 4,  # Fear
    74: 5,  # Surprise
    75: 1,  # Happy
    76: 1,  # Happy
    77: 5,  # Surprise
    78: 4,  # Fear
    79: 2,  # Sad
    80: 6   # Disgust
}

class DataHelper:
    @staticmethod
    def normalize_de_features(features):
        """
        Normalizuje cechy DE dla lepszego uczenia.

        Args:
            features: Tensor cech DE

        Returns:
            Tensor znormalizowanych cech
        """
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True) + 1e-8
        normalized = (features - mean) / std
        return normalized

    @staticmethod
    def load_mat_file(file_path, lds=False):
        """
        Wczytuje dane z pliku .mat.

        Args:
            file_path (str): Ścieżka do pliku .mat
            lds (bool): Czy filtrować tylko klucze zawierające "LDS"

        Returns:
            dict: Słownik zawierający dane z pliku .mat
        """
        mat_data = sio.loadmat(file_path)

        if lds:
            processed = {}
            for key in mat_data.keys():
                if contains(key, "LDS"):
                    processed[key] = mat_data[key]
            return processed
        return mat_data

    @staticmethod
    def print_mat_content(mat_data):
        """
        Wyświetla informacje o zawartości pliku .mat.

        Args:
            mat_data (dict): Słownik z danymi z pliku .mat
        """
        print("Klucze w pliku .mat:")
        for key in mat_data.keys():
            if not key.startswith('__'):  # Pomijamy klucze systemowe
                print(f"Klucz: {key}, Typ: {type(mat_data[key])}, Kształt: {mat_data[key].shape}")
        first = list(mat_data.values())[0]
        print(f"\nPrzykładowa zawartość klucza {list(mat_data.keys())[0]}:")

    @staticmethod
    def adapt_fft_to_original_format(data) -> EmotionDataset:
        """
        Wykonuje transformatę Fouriera a następnie dostosowuje wymiary,
        aby zachować format (x, 32) kompatybilny z resztą kodu.

        Args:
            data (dict): Słownik z danymi z pliku .mat

        Returns:
            EmotionDataset: Dataset z danymi FFT w formacie (x, 32)
        """
        import numpy as np
        from scipy.fftpack import fft

        # Indeksy elektrod do zachowania (32 z 62)
        indices = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
                   34, 36, 38, 40, 42, 44, 46, 48, 50, 53, 54, 55, 59]

        # Filtrujemy dane do wybranych elektrod
        processed_data = {}
        for key, array in data.items():
            processed_data[key] = array[:, :, indices]
            print(f"Klucz: {key}, Oryginalny kształt: {array.shape}, Nowy kształt: {processed_data[key].shape}")

        # Przygotowanie danych w formie listy słowników
        post_processed_data = []

        for key, array in processed_data.items():
            if contains(key, "LDS"):
                try:
                    # Ekstrakcja numeru wideo z klucza
                    number = int(key.split("_")[2])
                    # Mapowanie numeru wideo na emocję
                    emotion = VideoIdToEmotionMap[number]
                    if number:
                        # Dla każdego pomiaru w wymiarach (x, 5)
                        for i in range(array.shape[0]):
                            for j in range(array.shape[1]):
                                # Zastosuj transformatę Fouriera do danych z 32 elektrod
                                signal = array[i][j]

                                # Sprawdź, czy sygnał nie jest stały
                                if np.std(signal) < 1e-10:
                                    signal = signal + np.random.normal(0, 1e-5, size=signal.shape)

                                # Oblicz FFT i weź tylko część rzeczywistą (amplitudę)
                                # W przypadku sygnału rzeczywistego, druga połowa FFT jest lustrzanym odbiciem
                                fft_result = np.abs(fft(signal))[:len(signal)//2]

                                # Zachowaj te same wymiary - użyj pierwszych 32 komponentów FFT
                                if len(fft_result) >= 32:
                                    fft_features = fft_result[:32]
                                else:
                                    # Jeśli mamy mniej niż 32 komponenty, dopełnij zerami
                                    fft_features = np.zeros(32)
                                    fft_features[:len(fft_result)] = fft_result

                                # Dodaj wpis z cechami częstotliwościowymi
                                entry = {"emotion_id": emotion, "value": fft_features}
                                post_processed_data.append(entry)
                except (ValueError, IndexError) as e:
                    print(f"Błąd podczas przetwarzania {key}: {e}")
                    continue

        # Tworzenie datasetu
        dataset = EmotionDataset(post_processed_data)
        print(f"\nUtworzono dataset z {len(dataset.emotions)} próbkami.")
        print(f"Każda próbka zawiera {len(dataset.values[0])} cech częstotliwościowych.")

        # Dodajemy wizualizację korelacji między cechami częstotliwościowymi
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        # Konwertujemy dane do tablicy numpy
        values_array = np.array(dataset.values)

        # Sprawdź i usuń cechy ze stałymi wartościami (std=0)
        std_values = np.std(values_array, axis=0)
        valid_features = std_values > 1e-10

        if not np.all(valid_features):
            print(f"Znaleziono {np.sum(~valid_features)} cech ze stałą wartością (std=0).")
            print("Te cechy zostaną usunięte przed obliczeniem korelacji.")
            values_array_corr = values_array[:, valid_features]
        else:
            values_array_corr = values_array

        # Bezpieczne obliczanie macierzy korelacji
        try:
            values_df = pd.DataFrame(values_array_corr)
            corr_df = values_df.corr(method='pearson')
            corr_matrix = corr_df.fillna(0).values

            # Ustawienia wizualizacji
            plt.figure(figsize=(16, 14))

            # Tworzenie mapy ciepła
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)

            sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                        square=True, linewidths=.5, annot=False, fmt=".2f",
                        cbar_kws={"shrink": .7})

            plt.title('Mapa korelacji między cechami FFT (format 32)', fontsize=16)
            plt.tight_layout()
            plt.savefig('korelacja_fft_format32.png', dpi=300, bbox_inches='tight')

            # Statystyka korelacji
            triu_indices = np.triu_indices(corr_matrix.shape[0], k=1)
            correlation_values = corr_matrix[triu_indices]
            correlation_values = correlation_values[~np.isnan(correlation_values)]

            # Oblicz statystyki
            avg_corr = np.nanmean(correlation_values)
            max_corr = np.nanmax(correlation_values)
            min_corr = np.nanmin(correlation_values)

            print(f"\nAnaliza korelacji między cechami FFT (format 32):")
            print(f"Średnia korelacja: {avg_corr:.3f}")
            print(f"Minimalna korelacja: {min_corr:.3f}")
            print(f"Maksymalna korelacja: {max_corr:.3f}")
        except Exception as e:
            print(f"Błąd podczas wizualizacji korelacji: {e}")

        return dataset

    # Przykład użycia:
    # data = DataHelper.load_mat_file('sciezka_do_pliku.mat', lds=True)
    # dataset = DataHelper.adapt_fft_to_original_format(data)
    # tensor_dataset = dataset.tensorize()