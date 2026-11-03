"""Pipeline de teste rápido e leve para rodar localmente no notebook (CPU)"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import TestCNN


def run_test_pipeline():
    # 1. Configurações para teste rápido na máquina local
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Iniciando teste rápido ---")
    print(f"Dispositivo de execução: {device}")

    batch_size = 16
    max_samples = 64  # Usa apenas 64 amostras para execução rápida (ex: ~1 a 2 segundos)
    n_epochs = 1
    learning_rate = 0.001

    # 2. Dataloaders leves
    print(f"Carregando {max_samples} imagens...")
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        max_samples=max_samples,
        num_workers=0,      # num_workers=0 evita overhead de multiprocessing no teste
        pin_memory=False,
    )
    print(f"Batches de treino: {len(train_loader)} | Batches de validação: {len(val_loader)}")

    # 3. Modelo de teste
    model = TestCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Treinamento (1 época)
    model.train()
    total_train_loss = 0.0
    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * images.size(0)
        print(f"Treino | Batch [{step + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

    train_loss = total_train_loss / len(train_loader.dataset)
    print(f"Loss média de treino: {train_loss:.4f}")

    # 5. Validação
    model.eval()
    total_val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (preds == labels).sum().item()
            print(f"Validação | Batch [{step + 1}/{len(val_loader)}] - Loss: {loss.item():.4f}")

    val_loss = total_val_loss / len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)
    print(f"Loss média de validação: {val_loss:.4f} | Acurácia: {val_acc:.4f}")

    # 6. Teste de salvamento do modelo
    test_save_path = Path("test_model.pth")
    torch.save(model.state_dict(), test_save_path)
    print(f"Modelo de teste salvo com sucesso em: {test_save_path}")

    # Limpar arquivo temporário de teste se desejar
    if test_save_path.exists():
        test_save_path.unlink()
        print("Arquivo temporário 'test_model.pth' removido.")

    print("--- Teste do pipeline finalizado com sucesso! Tudo funcionando sem quebras. ---")


if __name__ == "__main__":
    run_test_pipeline()

