import random
from operator import contains
import scipy.io as sio
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np


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


class EEGWindowDataset(Dataset):
    def __init__(self, eeg_sequences, emotion_labels, window_size=128, stride=None, overlap=0.75):
        """
        Dataset do przetwarzania EEG z oknami czasowymi.

        Args:
            eeg_sequences: Lista tensorów o kształcie [time, bands, probes] dla każdego filmu
            emotion_labels: Lista etykiet emocji odpowiadających każdemu filmowi
            window_size: Rozmiar okna czasowego (liczba próbek)
            stride: Przesunięcie okna (jeśli None, używa wartości wyliczonej z overlap)
            overlap: Nakładanie się okien (używane tylko jeśli stride=None)
        """
        self.eeg_sequences = eeg_sequences
        self.emotion_labels = emotion_labels
        self.window_size = window_size

        if stride is None:
            self.stride = int(window_size * (1 - overlap))
        else:
            self.stride = stride

        # Przygotuj indeksy wszystkich możliwych okien
        self.window_indices = []
        for seq_idx, sequence in enumerate(eeg_sequences):
            seq_length = sequence.shape[0]  # time dimension

            # Wylicz wszystkie możliwe okna dla tej sekwencji
            for start_idx in range(0, seq_length - window_size + 1, self.stride):
                self.window_indices.append((seq_idx, start_idx))

        # Opcjonalnie: Możesz przetasować indeksy okien
        random.shuffle(self.window_indices)

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        seq_idx, start_idx = self.window_indices[idx]

        # Pobierz odpowiedni wycinek danych
        end_idx = start_idx + self.window_size
        window_data = self.eeg_sequences[seq_idx][start_idx:end_idx]

        # Zmień format na [bands, time, probes] dla modelu
        window_data = window_data.permute(1, 0, 2)

        # Pobierz etykietę dla tego filmu
        label = self.emotion_labels[seq_idx]

        return window_data, label


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

    @staticmethod
    def prepare_sequences_from_mat(mat_data):
        """
        Przygotowuje sekwencje EEG dla każdego filmu z pliku .mat.

        Args:
            mat_data: Dane z pliku .mat

        Returns:
            eeg_sequences: Lista sekwencji EEG w formacie [time, bands, probes]
            emotion_labels: Lista etykiet emocji
        """
        # Indeksy 32 elektrod do wyboru (z 62)
        selected_electrodes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                               26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50,
                               53, 54, 55, 59]

        eeg_sequences = []
        emotion_labels = []

        for key, array in mat_data.items():
            try:
                # Ekstrakcja emocji z kluczu
                video_number = int(key.split("_")[2] if "LDS" in key else key.split("_")[1])
                emotion = VideoIdToEmotionMap[video_number]

                # Wybór 32 elektrod z 62
                array = array[:, :, selected_electrodes]

                # Format: [time, bands, probes]
                eeg_sequences.append(torch.tensor(array, dtype=torch.float32))
                emotion_labels.append(emotion)

            except Exception as e:
                print(f"Błąd przetwarzania {key}: {e}")

        return eeg_sequences, emotion_labels

    @staticmethod
    def create_window_dataloader(mat_file_path, window_size=128, batch_size=32,
                                 stride=None, overlap=0.75, shuffle=True, num_workers=4):
        """
        Tworzy DataLoader z oknami czasowymi bezpośrednio z pliku .mat.

        Args:
            mat_file_path: Ścieżka do pliku .mat
            window_size: Rozmiar okna czasowego
            batch_size: Rozmiar batcha
            stride: Przesunięcie okna (jeśli None, używa overlap)
            overlap: Nakładanie się okien (używane tylko jeśli stride=None)
            shuffle: Czy mieszać dane
            num_workers: Liczba wątków do ładowania danych

        Returns:
            DataLoader: DataLoader dla okien czasowych
        """
        # Wczytaj dane
        mat_data = DataHelper.load_mat_file(mat_file_path)

        # Przygotuj sekwencje
        eeg_sequences, emotion_labels = DataHelper.prepare_sequences_from_mat(mat_data)

        # Stwórz dataset
        dataset = EEGWindowDataset(
            eeg_sequences,
            emotion_labels,
            window_size=window_size,
            stride=stride,
            overlap=overlap
        )

        # Stwórz dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    @staticmethod
    def create_cv_datasets(mat_file_path, num_folds=5, window_size=128,
                           stride=None, overlap=0.75):
        """
        Tworzy zestawy danych do walidacji krzyżowej, gdzie każdy fold
        zawiera inne filmy jako zestaw testowy.

        Args:
            mat_file_path: Ścieżka do pliku .mat
            num_folds: Liczba foldów walidacji krzyżowej
            window_size: Rozmiar okna czasowego
            stride: Przesunięcie okna
            overlap: Nakładanie się okien

        Returns:
            folds: Lista tupli (train_loader, val_loader) dla każdego foldu
        """
        # Wczytaj dane
        mat_data = DataHelper.load_mat_file(mat_file_path)

        # Przygotuj sekwencje
        eeg_sequences, emotion_labels = DataHelper.prepare_sequences_from_mat(mat_data)

        # Liczba sekwencji
        num_sequences = len(eeg_sequences)

        # Indeksy sekwencji
        sequence_indices = list(range(num_sequences))

        # Pomieszaj indeksy
        np.random.shuffle(sequence_indices)

        # Podziel indeksy na foldy
        fold_size = num_sequences // num_folds
        folds = []

        for fold_idx in range(num_folds):
            # Indeksy dla zestawu testowego
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size if fold_idx < num_folds - 1 else num_sequences
            val_indices = sequence_indices[val_start:val_end]

            # Indeksy dla zestawu treningowego
            train_indices = [idx for idx in sequence_indices if idx not in val_indices]

            # Przygotuj zestawy danych
            train_sequences = [eeg_sequences[idx] for idx in train_indices]
            train_labels = [emotion_labels[idx] for idx in train_indices]

            val_sequences = [eeg_sequences[idx] for idx in val_indices]
            val_labels = [emotion_labels[idx] for idx in val_indices]

            # Stwórz datasety
            train_dataset = EEGWindowDataset(
                train_sequences, train_labels,
                window_size=window_size, stride=stride, overlap=overlap
            )

            val_dataset = EEGWindowDataset(
                val_sequences, val_labels,
                window_size=window_size, stride=stride, overlap=overlap
            )

            # Stwórz dataloadery
            train_loader = DataLoader(
                train_dataset, batch_size=32, shuffle=True,
                num_workers=4, pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset, batch_size=32, shuffle=False,
                num_workers=4, pin_memory=True
            )

            folds.append((train_loader, val_loader))

        return folds

# Przykład użycia:
# 1. Utworzenie data loadera z oknami czasowymi
# window_loader = DataHelper.create_window_dataloader(
#     'sciezka_do_pliku.mat',
#     window_size=128,
#     batch_size=32,
#     overlap=0.75
# )
#
# 2. Utworzenie zbiorów do walidacji krzyżowej
# cv_folds = DataHelper.create_cv_datasets(
#     'sciezka_do_pliku.mat',
#     num_folds=5,
#     window_size=128,
#     overlap=0.75
# )
#
# 3. Iteracja po batchu danych
# for window_batch, labels in window_loader:
#     # window_batch ma format [batch, bands, time, probes]
#     outputs = model(window_batch)
#     # ...