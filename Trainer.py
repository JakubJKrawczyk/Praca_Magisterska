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
            for i, (labels, inputs) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zerowanie gradientów
                optimizer.zero_grad()
                
                # Forward pass - przetwarzamy cały batch naraz
                outputs = model(inputs)  # Powinno zwrócić [batch_size, num_classes]
                
                # Oblicz stratę dla całego batcha
                loss = criterion(outputs, labels)
                
                # Backward pass i optymalizacja
                loss.backward()
                optimizer.step()
                
                # Statystyki
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                self.logger.display_progress(i, len(train_loader), "Training...", 
                                            f"Batch {i+1}/{len(train_loader)} Epoch {epoch+1}/{num_epochs} "
                                            f"loss: {loss.item():.4f} Previous epoch loss: {epoch_loss:.5f} "
                                            f"Accuracy: {epoch_accuracy:.5f}%")
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

                self.logger.display_progress(total // labels.size(0), len(test_loader), "Evaluating...", f"Batch {total // labels.size(0)}/{len(test_loader)}")
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

