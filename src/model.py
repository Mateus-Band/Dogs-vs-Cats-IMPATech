"""Arquitetura da rede convolucional (V4 - 5 Blocos para 384x384)"""

import torch
import torch.nn as nn


# classe para teste leve (rodar no notebook)
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

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # classificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),  # saída binária
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# Modelo grande final (V4 - 384x384 com 5 Blocos Convolucionais)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Extrator de Características Profundo
        # Entrada: (3, 384, 384)
        self.features = nn.Sequential(
            # Bloco 1: 3 -> 32 | Mapa: 192x192
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Bloco 2: 32 -> 64 | Mapa: 96x96
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Bloco 3: 64 -> 128 | Mapa: 48x48
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Bloco 4: 128 -> 256 | Mapa: 24x24
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Bloco 5 (NOVO): 256 -> 512 | Mapa: 12x12
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Global Average Pooling: Converte (512, 12, 12) em (512, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classificador: 512 -> Dropout(0.5) -> Linear(512, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),  # Logit binário
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 384, 384)
    model = CNN()
    print("Saída CNN V4:", model(x).shape)
