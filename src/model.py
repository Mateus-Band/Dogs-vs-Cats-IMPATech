"""Arquitetura da rede convolucional"""

import torch
import torch.nn as nn


#classe para teste leve (rodar no notebook)
class TestCNN(nn.Module):
    # init para declaração das camadas
    def __init__(self):
        super().__init__()
        # feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # classificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),  # saída binária
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Modelo grande final (maquina impatech)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Extrator de Características Profundo
        # Entrada: (3, 224, 224)
        self.features = nn.Sequential(
            # Bloco 1: Output 224x224 -> MaxPool -> 112x112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Bloco 2: Output 112x112 -> MaxPool -> 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Bloco 3: Output 56x56 -> MaxPool -> 28x28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Bloco 4: Output 28x28 -> MaxPool -> 14x14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Global Average Pooling: Converte (512, H, W) em (512, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # Logit binário
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)  # Lote fictício de 2 imagens 224x224
    test_model = TestCNN()
    print("Saída TestCNN:", test_model(x).shape)  # Deve imprimir torch.Size([2, 1])