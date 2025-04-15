from operator import contains, indexOf

import scipy.io as sio
import torch
from pandas.core.interchange.utils import dtype_to_arrow_c_fmt
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Tworzymy własną klasę Dataset, która zachowa powiązanie
# między etykietami a wartościami
class EmotionDataset(Dataset):
    def __init__(self, emotion_data:[{int, np.ndarray}]):
        self.emotions = []
        self.values = []
        self.idx_to_emotion = {}
        # print(emotion_data[0].keys())
        # Przypisujemy indeksy do emocji
        for idx, emotion_array in enumerate(emotion_data):
            self.idx_to_emotion[idx] = emotionsEnum[emotion_array["emotion_id"]]

        # Wypełniamy listy danych
        for emotion_array in emotion_data:
            self.emotions.append(emotion_array["emotion_id"])
            self.values.append(emotion_array["value"])

    def tensorize(self):
        # Konwertujemy na tensory
        emotions_tensor = torch.tensor(self.emotions, dtype=torch.int32)
        values_tensor = torch.tensor(self.values, dtype=torch.float)
        tensor = TensorDataset(emotions_tensor, values_tensor)
        return tensor

    def __len__(self):
        return len(self.emotions)

    def get_emotion(self, idx):
        return self.idx_to_emotion[idx]

    def extend(self, other):
        if isinstance(other, EmotionDataset):
            self.emotions.extend(other.emotions)
            self.values.extend(other.values)
            incrementer = len(self.idx_to_emotion)
            for emotion in other.idx_to_emotion.values():
                self.idx_to_emotion[incrementer] = emotion
                incrementer += 1

        else:
            raise TypeError("Can only add another EmotionDataset instance.")

emotionsEnum = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Anger",
    4: "Fear",
    5: "Surprise",
    6: "Disgust"
}

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
    # Wczytywanie pliku .mat
    def load_mat_file(file_path, lds=False):
        mat_data = sio.loadmat(file_path)

        if lds:
            processed = {}
            for key in mat_data.keys():
                if contains(key, "LDS"):
                    processed[key] = mat_data[key]
            return processed
        return mat_data




    @staticmethod
    # Wyświetlanie zawartości pliku .mat
    def print_mat_content(mat_data):
        print("Klucze w pliku .mat:")
        for key in mat_data.keys():
            if not key.startswith('__'):  # Pomijamy klucze systemowe
                 print(f"Klucz: {key}, Typ: {type(mat_data[key])}, Kształt: {mat_data[key].shape}")
        first = list(mat_data.values())[0]
        print(f"\nPrzykładowa zawartość klucza {first}:")
    # Prztworzyć dane od sond tylko:
    # 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
    # 34, 36, 38, 40, 42, 44, 46, 48, 50, 53, 54, 55, 59, 60, 61

    @staticmethod
    # przetwarzanie wczytanych danych pod model
    # dane w formie (x, 5, 62) przerobić na (x,5,32)
    def prepare_data(data) -> EmotionDataset:
        # DataHelper.print_mat_content(data)
        indices = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
                   34, 36, 38, 40, 42, 44, 46, 48, 50, 53, 54, 55, 59]

        processed_data = {}
        for key, array in data.items():
            processed_data[key] = array[:, :, indices]
            # print(f"Klucz: {key}, Nowy kształt: {processed_data[key].shape}")
            # print(processed_data[key][0])

        post_processed_data = []
        for key, array in processed_data.items():
            if contains(key, "LDS"):
                try:
                    number = int(key.split("_")[2])
                    emotion = VideoIdToEmotionMap[number]
                    if number:
                        for i in range(array.shape[0]):
                            for j in range(array.shape[1]):
                                entry = {"emotion_id": emotion, "value": array[i][j]}
                                post_processed_data.append(entry)
                except (ValueError, IndexError):
                    continue

        dataset = EmotionDataset(post_processed_data)
        # Przykład użycia:
        # dataset = przetworz_dane_do_torch(data, VideoIdToEmotionMap, emotionsEnum)
        # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Przykładowe dalsze przetwarzanie
        return dataset


