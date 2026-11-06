"""Loop de treinamento"""

from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import ROOT_DIR, get_dataloaders
from model import CNN

# Hiperparametros
BATCH_SIZE = 48
LEARNING_RATE = 0.001
ETA_MIN = 1e-6
WEIGHT_DECAY = 1e-4
N_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

OUTPUT_DIR = ROOT_DIR / "data"


def train_model():
    # configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando em: {device}\n")

    batch_size = BATCH_SIZE
    lr = LEARNING_RATE
    eta_min = ETA_MIN
    weight_decay = WEIGHT_DECAY
    n_epochs = N_EPOCHS
    patience = EARLY_STOPPING_PATIENCE

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "best_model_3.0.pth"
    history_path = OUTPUT_DIR / "training_history_3.0.csv"
    val_preds_path = OUTPUT_DIR / "val_predictions_3.0.csv"

    # modelo e dataloader
    train_loader, val_loader = get_dataloaders(batch_size)
    model = CNN().to(device)

    # otimização, perda e scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=eta_min)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    # Loop de Epochs
    for epoch in range(n_epochs):
        # TREINO:
        model.train()
        train_loss, train_correct = 0.0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{n_epochs}] Treino")
        for images, labels in train_bar:
            images = images.to(device)
            # BCEWithLogitsLoss espera rótulo como (N, 1) float
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
        val_true_list, val_prob_list, val_pred_list = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1).float()

                outputs = model(images)
                loss = criterion(outputs, labels)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                val_loss += loss.item() * images.size(0)
                val_correct += (preds == labels).sum().item()

                val_true_list.extend(labels.squeeze(1).cpu().numpy().tolist())
                val_prob_list.extend(probs.squeeze(1).cpu().numpy().tolist())
                val_pred_list.extend(preds.squeeze(1).cpu().numpy().tolist())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / len(val_loader.dataset)
        current_lr = optimizer.param_groups[0]["lr"]

        # Passo do scheduler ao final da época
        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{n_epochs}] (LR: {current_lr:.6f})\n"
            f"  Treino Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f}\n"
            f"  Val Loss:    {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f}"
        )

        # Registrar histórico da época e salvar em disco
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
            }
        )
        pd.DataFrame(history).to_csv(history_path, index=False)

        # Salvar melhor modelo e as previsões de validação correspondentes
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            pd.DataFrame(
                {
                    "y_true": val_true_list,
                    "y_prob": val_prob_list,
                    "y_pred": val_pred_list,
                }
            ).to_csv(val_preds_path, index=False)
            print(f"  -> Novo melhor modelo salvo em: {save_path}")
        else:
            epochs_without_improvement += 1
            print(f"  -> Sem melhora na Val Loss ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print(f"\nEarly Stopping acionado após {patience} épocas sem melhora na Val Loss.")
                break

    print(f"\nHistórico de treinamento salvo em: {history_path}")
    print(f"Previsões de validação do melhor modelo salvas em: {val_preds_path}")


if __name__ == "__main__":
    train_model()
