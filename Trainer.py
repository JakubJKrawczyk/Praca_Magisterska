import os
from time import sleep

import torch

from Custom.Data.Logger import Logger


class Trainer:
    def __init__(self,device):
        self.device = device
        self.logger = Logger()
    def train_model(self, model, train_loader, criterion, optimizer, num_epochs=10):
        self.logger.display_progress(0, num_epochs, "Starting training...")
        sleep(2)
        model.train()
        train_losses = []
        train_accuracies = []

        for epoch in range(num_epochs):
            self.logger.display_progress(epoch, num_epochs, "Training...", f"Epoch {epoch+1}/{num_epochs}")
            running_loss = 0.0
            correct = 0
            total = 0
            epoch_loss = 0.0
            epoch_accuracy = 0.0

            # Poprawiona kolejność rozpakowania danych z data loadera
            for i, (labels,inputs) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Iteracja po każdym przykładzie w batchu
                for j in range(labels.size(0)):
                    # Pobierz pojedynczy przykład
                    single_input = inputs[j]  # Tensor o kształcie [32, d_model]
                    single_label = labels[j]  # Skalar, np. 0, 1, 2...

                    # Zerowanie gradientów
                    optimizer.zero_grad()

                    # Forward pass
                    output = model(single_input)  # Powinno zwrócić [7] - logity dla każdej klasy

                    # Przygotuj etykietę w formie wymaganej przez criterion
                    # Przekształć skalar na tensor o wymiarze [1]
                    target = single_label.long()  # Konwersja na Long

                    # Dodaj wymiar batcha do output jeśli potrzeba
                    if len(output.shape) == 1:  # Jeśli output ma kształt [7]
                        output = output.unsqueeze(0)  # Zmień na [1, 7]

                    # Sprawdź kształty przed obliczeniem straty

                    # Oblicz stratę
                    loss = criterion(output, target.unsqueeze(0))

                    # Backward pass i optymalizacja
                    loss.backward()
                    optimizer.step()
                    print(f"Loss: {loss.item()}")
                    # Statystyki
                    running_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += 1
                    if predicted == target:
                        correct += 1

                    self.logger.display_progress(j, inputs.size(0), "Training...", f"Sample {j+1}/{inputs.size(0)} Epoch {epoch+1}/{num_epochs} loss: {loss.item():.4f} Previous epoch loss: {epoch_loss:.5f} Accuracy: {epoch_accuracy:.5f}%")
                # Wyświetl postęp po każdym batchu

                print(f'Epoka [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

            epoch_loss = running_loss / total
            epoch_accuracy = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            print(f'Epoka [{epoch+1}/{num_epochs}], Średnia strata: {epoch_loss:.4f}, '
                  f'Dokładność: {epoch_accuracy:.2f}%')

        self.logger.display_progress(num_epochs, num_epochs, "Finishing training...")
        sleep(1)
        return train_losses, train_accuracies

    # Funkcja do ewaluacji modelu
    def evaluate_model(self, model, test_loader, criterion):

        self.logger.display_progress(0, len(test_loader), "Starting evaluation...")
        sleep(1)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for labels, inputs in test_loader:

                self.logger.display_progress(test_loader.indexOf(labels), len(test_loader), "Evaluating...", f"Batch {test_loader.indexOf(labels)}/{len(test_loader)}")
                sleep(0.5)

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.logger.display_progress(len(test_loader), len(test_loader), "Evaluating...", f"Counting accuracy...")
        sleep(0.5)

        accuracy = 100 * correct / total
        avg_loss = test_loss / len(test_loader)
        print(f'Ewaluacja - Strata: {avg_loss:.4f}, Dokładność: {accuracy:.2f}%')

        return avg_loss, accuracy

