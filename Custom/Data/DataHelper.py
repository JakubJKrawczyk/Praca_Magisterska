from operator import contains

import scipy.io as sio

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