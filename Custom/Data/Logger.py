import os
import sys
import time

class Logger:
    def __init__(self, bar_length=50, bar_char="█", empty_char="░"):
        """
        Inicjalizacja loggera z paskiem postępu.

        Args:
            bar_length (int): Długość paska postępu w znakach
            bar_char (str): Znak używany do wypełnienia paska
            empty_char (str): Znak używany do pustych miejsc paska
        """
        self.bar_length = bar_length
        self.bar_char = bar_char
        self.empty_char = empty_char

    def clear_console(self):
        """Czyści konsolę w zależności od systemu operacyjnego."""
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Linux, macOS, etc.
            os.system('clear')

    def display_progress(self, current, maximum, header="Postęp", additional_text=""):
        """
        Wyświetla pasek postępu z nagłówkiem i dodatkowym tekstem.

        Args:
            current (float): Aktualna wartość
            maximum (float): Maksymalna wartość
            header (str): Nagłówek wyświetlany przed paskiem
            additional_text (str): Dodatkowy tekst wyświetlany pod paskiem
        """
        self.clear_console()

        # Zabezpieczenie przed wartościami ujemnymi i dzieleniem przez zero
        current = max(0, current)
        maximum = max(1, maximum)
        current = min(current, maximum)

        # Obliczanie procentu ukończenia
        percent = current / maximum

        # Tworzenie paska postępu
        filled_length = int(self.bar_length * percent)
        bar = self.bar_char * filled_length + self.empty_char * (self.bar_length - filled_length)

        # Wyświetlanie całości
        print(f"{header}")
        print(f"[{bar}] {percent:.1%} ({current}/{maximum})")

        if additional_text:
            print(f"\n{additional_text}")

        # Zapewnienie natychmiastowego wyświetlenia (bez buforowania)
        sys.stdout.flush()
