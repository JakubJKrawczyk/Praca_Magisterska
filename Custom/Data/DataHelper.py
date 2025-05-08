from operator import contains
import scipy.io as sio
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np

class EmotionDataset(Dataset):
    """
    Dataset przechowujący dane EEG wraz z etykietami emocji.

    Każdy element datasetu to para (emotion_id, electrode_values), gdzie:
    - emotion_id: identyfikator emocji (0-6)
    - electrode_values: słownik zawierający wartości 5 pasm (delta, theta, alpha, beta, gamma)
                         dla 32 elektrod
    """
    def __init__(self, emotion_data):
        self.emotions = []
        self.values = []
        self.idx_to_emotion = {}

        # Przypisujemy indeksy do emocji
        for idx, emotion in enumerate(emotionsEnum.values()):
            self.idx_to_emotion[idx] = emotion

        # Wypełniamy listy danych
        for emotion_array in emotion_data:
            self.emotions.append(emotion_array["emotion_id"])
            self.values.append(emotion_array["value"])

        # Wyświetlenie informacji o danych
        if len(self.values) > 0:
            print(f"EmotionDataset created with {len(self.emotions)} samples")
            print(f"Each sample contains values for 5 bands and 32 electrodes")

    def tensorize(self):
        """
        Konwertuje dane na tensory PyTorch i zwraca TensorDataset.

        Returns:
            TensorDataset: Dataset zawierający parę tensorów (values, emotions)
                emotions: tensor o wymiarze (n_samples,) - identyfikatory emocji
                values: tensor o wymiarze (n_samples, 5, 32) - wartości z 5 pasm dla 32 elektrod
        """
        # Konwertujemy emocje na tensor
        emotions_tensor = torch.tensor(self.emotions, dtype=torch.long)

        # Konwertujemy słowniki pasm na tensor 3D [n_samples, 5, 32]
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
        values_np = np.zeros((len(self.values), len(bands), 32), dtype=np.float32)

        for i, sample in enumerate(self.values):
            for j, band in enumerate(bands):
                values_np[i, j] = np.array(sample[band], dtype=np.float32)

        values_tensor = torch.tensor(values_np, dtype=torch.float)

        # Normalizacja
        values_tensor = DataHelper.normalize_de_features(values_tensor)

        # Sprawdzenie wymiarów
        print(f"Tensors created: emotions {emotions_tensor.shape}, values {values_tensor.shape}")

        # Tworzenie TensorDataset
        return TensorDataset(values_tensor, emotions_tensor)

    def __len__(self):
        return len(self.emotions)

    def get_emotion(self, idx):
        return self.idx_to_emotion[self.emotions[idx]]

    def extend(self, other):
        """
        Rozszerza dataset o dane z innego EmotionDataset lub listy słowników.

        Args:
            other: Może być:
                - EmotionDataset: Inna instancja EmotionDataset
                - list: Lista słowników w formacie [{emotion_id: id, value: {delta: value, theta: value, ...}}]
        """
        if isinstance(other, EmotionDataset):
            self.emotions.extend(other.emotions)
            self.values.extend(other.values)
        elif isinstance(other, list):
            # Sprawdzamy czy lista zawiera słowniki w odpowiednim formacie
            try:
                for item in other:
                    if "emotion_id" in item and "value" in item:
                        # Weryfikacja kluczy we values
                        required_bands = ["delta", "theta", "alpha", "beta", "gamma"]
                        if all(band in item["value"] for band in required_bands):
                            self.emotions.append(item["emotion_id"])
                            self.values.append(item["value"])
                        else:
                            missing_bands = [band for band in required_bands if band not in item["value"]]
                            raise ValueError(f"Missing bands in item's value: {missing_bands}")
                    else:
                        raise ValueError("Each item must contain 'emotion_id' and 'value' keys")

                # Po przetworzeniu wyświetl informację o liczbie dodanych elementów
                print(f"Added {len(other)} samples to dataset")

            except (KeyError, ValueError) as e:
                raise ValueError(f"Invalid format in the list: {str(e)}")
        else:
            raise TypeError("Can only add another EmotionDataset instance or a list of properly formatted dictionaries")


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

# positiveNegative_enum = {
#     0: "Negative",
#     1: "Positive"
# }

# def mapuj_emocje(liczba : int, return_id : bool):
#     if liczba in [1, 5, 0]:
#         if return_id:
#             return 1
#         else:
#             return positiveNegative_enum[1]
#     elif liczba in [2, 3, 4, 6]:
#         if return_id:
#             return 0
#         else:
#             return positiveNegative_enum[0]

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
        Normalizuje cechy dla lepszego uczenia.

        Args:
            features: Tensor cech w formacie [samples, bands, electrodes]

        Returns:
            Tensor znormalizowanych cech
        """
        # Normalizacja dla każdej próbki oddzielnie
        # Reshape do [samples, bands*electrodes]
        batch_size = features.shape[0]
        flat_features = features.reshape(batch_size, -1)

        # Normalizacja
        mean = flat_features.mean(dim=1, keepdim=True)
        std = flat_features.std(dim=1, keepdim=True) + 1e-8
        normalized = (flat_features - mean) / std

        # Reshape z powrotem do [samples, bands, electrodes]
        return normalized.reshape(features.shape)

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
        else:
            processed = {}
            for key in mat_data.keys():
                if contains(key, "de") and not contains(key, "LDS") and not contains(key, "__header__"):
                    processed[key] = mat_data[key]
            return processed

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
    def adapt_to_emotion_format(data) -> list:
        """
        Konwertuje dane w formacie [próbki, pasma, elektrody] na docelową strukturę.
        """
        # Indeksy 32 elektrod do wyboru (z 62)
        selected_electrodes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                               26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50,
                               53, 54, 55, 59]

        # Mapowanie pasm na nazwy
        bands = ["delta", "theta", "alpha", "beta", "gamma"]

        processed_data = []

        for key, array in data.items():
            try:
                # Ekstrakcja emocji z klucza
                video_number = int(key.split("_")[2] if "LDS" in key else key.split("_")[1])
                emotion = VideoIdToEmotionMap[video_number]

                # Wybór 32 elektrod z 62
                array = array[:, :, selected_electrodes]

                # Dla każdej próbki czasowej
                for sample_idx in range(array.shape[0]):
                    band_data = {}

                    # Dla każdego pasma
                    for band_idx, band_name in enumerate(bands):
                        # Pobierz wartości dla 32 elektrod
                        electrodes_values = array[sample_idx, band_idx, :].tolist()
                        band_data[band_name] = electrodes_values

                    processed_data.append({
                        "emotion_id": emotion,
                        "value": band_data
                    })

            except Exception as e:
                print(f"Błąd przetwarzania {key}: {e}")

        return processed_data

# Przykład użycia:
# 1. Wczytanie danych z pliku .mat
# mat_data = DataHelper.load_mat_file('sciezka_do_pliku.mat', lds=True)
#
# 2. Konwersja danych do odpowiedniego formatu
# processed_data = DataHelper.adapt_to_emotion_format(mat_data)
#
# 3. Tworzenie datasetu
# dataset = EmotionDataset(processed_data)
#
# 4. Konwersja na tensory
# tensor_dataset = dataset.tensorize()
#
# 5. Utworzenie dataloaderaL
# dataloader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)