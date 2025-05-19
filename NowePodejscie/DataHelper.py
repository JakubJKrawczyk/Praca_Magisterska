import os
from operator import contains

import mne
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
    80: 6  # Disgust
}


class FilmWindowDataset(Dataset):
    def __init__(self, eeg_sequences_per_movie, emotion_labels, first_win_size=16, stride = 8, is_raw=False):
        """
        Dataset tworzący tensor okien czasowych dla każdego filmu z przesunięciem

        Args:
            eeg_sequences_per_movie: Lista tensorów o kształcie [samples, bands, probes] albo [samples, raw_values] dla każdego filmu
            emotion_labels: Lista etykiet emocji odpowiadających każdemu filmowi
            first_win_size: rozmiar pierwszego okna
            overlap: Nakładanie się okien, zamienione na int (domyślnie 20%)
        """
        self.eeg_sequences = eeg_sequences_per_movie
        self.emotion_labels = emotion_labels
        self.first_window_size = first_win_size
        self.stride = stride
        self.is_raw = is_raw
        self.data_mean = np.mean(eeg_sequences_per_movie)

        print(f"Pierwsze okno ma rozmiar {self.first_window_size} próbek z przesunięciem {self.stride} próbek")

        # Przygotuj indeksy filmów i oblicz liczbę okien dla każdego filmu
        self.film_indices = []
        self.windows_per_film = []
        for movie_idx, sequence in enumerate(self.eeg_sequences):
            window_step = self.first_window_size
            self.windows_per_film[movie_idx] = 0
            while window_step < len(eeg_sequences_per_movie):
                self.windows_per_film[movie_idx] += 1
                window_step += self.stride
            if len(eeg_sequences_per_movie[window_step:len(self.eeg_sequences)]) > 0:
                self.windows_per_film[movie_idx] += 1
            self.film_indices.append(movie_idx)

        total_windows = sum(self.windows_per_film)
        print(f"Załadowano {len(self.film_indices)} filmów z łączną liczbą {total_windows} okien")

    def __len__(self):
        # Zwraca liczbę filmów (nie okien)
        return len(self.film_indices)

    def get_film_windows(self, film_idx):
        """
        Zwraca wszystkie okna dla danego filmu jako batch.

        Args:
            film_idx: Indeks filmu

        Returns:
            Tensor okien o kształcie [num_windows, bands, window_size, probes]
            Etykieta emocji dla tego filmu
        """

        sequence = self.eeg_sequences[film_idx]
        seq_length = sequence.shape[0]
        data_mean = np.mean(sequence)

        # Lista okien dla tego filmu
        windows = []
        number_of_windows = self.windows_per_film[film_idx]
        # Generuj okna z odpowiednim nakładaniem
        for window_ids in range(number_of_windows):

            window_step = self.first_window_size
            # jezeli ostatnie okno jest niepelne to wypelnij je wartosciami usrednionymi
            if window_step > seq_length-1:
                for new_val_index in range(window_step - seq_length-1):
                    sequence[new_val_index] = data_mean

            # Wytnij okno czasowe
            window = sequence[window_step - self.first_window_size:window_step]

            windows.append(window)
            window_step += self.stride

        # Stack wszystkich okien dla danego filmu
        if windows:
            film_windows = torch.stack(windows)
        else:
            raise Exception("Nieudana próba utworzenia okien dla filmu o podanym indeksie")

        # Zwróć okna i etykietę dla filmu
        return film_windows, self.emotion_labels[film_idx]

    def __getitem__(self, idx):
        """
        Zwraca wszystkie okna dla danego filmu jako pojedynczy element.

        Args:
            idx: Indeks filmu (nie okna)

        Returns:
            Tensor okien dla filmu i etykietę emocji
        """
        if idx > len(self.film_indices) or idx < 0:
            raise ValueError(f"Film {idx}  doesn't exist")

        film_idx = self.film_indices[idx]
        return self.get_film_windows(film_idx)


class DataHelper:
    @staticmethod
    def load_processed_data_from_mat_file(file_path, lds=False):
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
    def load_raw_data_from_file(file_path):
        # Load the CNT file
        raw = mne.io.read_raw_cnt(file_path, preload=True)

        # Display basic information about the dataset
        print(raw.info)

        # Plot the raw data (optional)
        raw.plot()

        # Access the data as a NumPy array
        data, times = raw.get_data(return_times=True)

        print(data.shape)

    @staticmethod
    def prepare_data(data_dir_file):
        files = os.listdir(data_dir_file)
        data = []
        data_to_emotion_idx = []
        if all(s.endswith(".mat") for s in files):
            for idx, mat in enumerate(files):
                data.append(DataHelper.load_processed_data_from_mat_file(data_dir_file + mat))
        elif all(s.endswith(".cnt") for s in files):
            for cnt in files:
                data.append(DataHelper.load_raw_data_from_file(data_dir_file + cnt))
        else:
            raise ValueError("Obecny rodzaj rozszerzenia w docelowym folderze jest nieobsługiwany")






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
            video_numbers: Lista numerów filmów
        """
        # Indeksy 32 elektrod do wyboru (z 62)
        selected_electrodes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                               26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50,
                               53, 54, 55, 59]

        eeg_sequences = []
        emotion_labels = []
        video_numbers = []

        for key, array in mat_data.items():
            try:
                # Ekstrakcja numeru filmu z klucza
                video_number = int(key.split("_")[2] if "LDS" in key else key.split("_")[1])
                emotion = VideoIdToEmotionMap[video_number]

                # Wybór 32 elektrod z 62
                array = array[:, :, selected_electrodes]

                # Format: [time, bands, probes]
                eeg_sequences.append(torch.tensor(array, dtype=torch.float32))
                emotion_labels.append(emotion)
                video_numbers.append(video_number)

                print(f"Przetworzono film {video_number}: {array.shape[0]} próbek, emocja: {emotionsEnum[emotion]}")

            except Exception as e:
                print(f"Błąd przetwarzania {key}: {e}")

        return eeg_sequences, emotion_labels, video_numbers

    @staticmethod
    def create_film_window_dataloader(mat_file_path, num_windows=5, batch_size=1,
                                      shuffle=True, num_workers=4):
        """
        Tworzy DataLoader zwracający batche okien dla każdego filmu, zapewniając,
        że najkrótszy film ma dokładnie num_windows okien.

        Args:
            mat_file_path: Ścieżka do pliku .mat
            num_windows: Liczba okien dla najkrótszego filmu (domyślnie 5)
            batch_size: Ile filmów w batchu (zalecane 1)
            shuffle: Czy mieszać filmy
            num_workers: Liczba wątków do ładowania danych

        Returns:
            DataLoader: DataLoader dla filmów z oknami
        """
        # Wczytaj dane
        mat_data = DataHelper.load_mat_file(mat_file_path)

        # Przygotuj sekwencje
        eeg_sequences, emotion_labels, video_numbers = DataHelper.prepare_sequences_from_mat(mat_data)

        # Stwórz dataset
        dataset = FilmWindowDataset(
            eeg_sequences,
            emotion_labels,
            first_win_size=num_windows,  # Najkrótszy film będzie miał 5 okien
            overlap=0.5  # 50% nakładania się
        )

        # Stwórz dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,  # Zwykle 1, bo każdy film ma wiele okien
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    @staticmethod
    def create_cv_datasets_per_film(mat_file_path, num_folds=5, num_windows=5, overlap=0.5):
        """
        Tworzy zestawy danych do walidacji krzyżowej, gdzie każdy fold
        zawiera inne filmy jako zestaw testowy.

        Args:
            mat_file_path: Ścieżka do pliku .mat
            num_folds: Liczba foldów walidacji krzyżowej
            num_windows: Liczba okien dla najkrótszego filmu (domyślnie 5)
            overlap: Nakładanie się okien (50%)

        Returns:
            folds: Lista tupli (train_loader, val_loader) dla każdego foldu
        """
        # Wczytaj dane
        mat_data = DataHelper.load_mat_file(mat_file_path)

        # Przygotuj sekwencje
        eeg_sequences, emotion_labels, video_numbers = DataHelper.prepare_sequences_from_mat(mat_data)

        # Liczba sekwencji (filmów)
        num_sequences = len(eeg_sequences)

        # Indeksy filmów
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
            train_dataset = FilmWindowDataset(
                train_sequences, train_labels,
                first_win_size=num_windows, overlap=overlap
            )

            val_dataset = FilmWindowDataset(
                val_sequences, val_labels,
                first_win_size=num_windows, overlap=overlap
            )

            # Stwórz dataloadery
            train_loader = DataLoader(
                train_dataset, batch_size=1, shuffle=True,
                num_workers=4, pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset, batch_size=1, shuffle=False,
                num_workers=4, pin_memory=True
            )

            folds.append((train_loader, val_loader))

        return folds