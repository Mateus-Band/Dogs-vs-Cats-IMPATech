"""Loop de treinamento"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import get_dataloaders
from model import CNN

# Hiperparametros
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_EPOCHS = 15



def train_model():
    # configs
    device = torch.device("cuda")
    print(f"Treinando em: {device}\n")

    batch_size = BATCH_SIZE
    lr = LEARNING_RATE
    n_epochs = N_EPOCHS

    # modelo e dataloader
    train_loader, val_loader = get_dataloaders(batch_size)
    model = CNN().to(device)

    # otimização e perca
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    save_path = Path("best_model.pth")

    # Loop de Epochs
    for epoch in range(n_epochs):
        # treino:
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images = images.to(device)
            # BCEWithLogis... bla bla espera rotulo como (N, 1)
            labels = labels.to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (preds == labels).sum().item()
    
    epoch_train_loss = train_loss / len(train_loader.dataset)
    epoch_train_acc = train_correct / len(train_loader.dataset)
    
    # VALIDAÇÃO:
    model.eval()
    val_loss, val_correct = 0.0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (preds == labels).sum().item()

    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc = val_correct / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{n_epochs}]\n Treino Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}\nVal Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

    # salvar melhor modelo
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Novo melhor modelo salvo: {save_path}")


if __name__ == "__main__":
    train_model()