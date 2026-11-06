"""Loop de Fine-Tuning da V4 (4.0_Final)"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import ROOT_DIR, get_dataloaders, get_finetune_transform
from model import CNN

# Hiperparâmetros do Fine-Tuning
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
ETA_MIN = 1e-7
WEIGHT_DECAY = 1e-4
N_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 4
ROTATION_DEGREES = 5

OUTPUT_DIR = ROOT_DIR / "data"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_initial_blocks(model: CNN) -> None:
    """
    Congela do Bloco 1 ao Bloco 3 (requires_grad = False) e mantém
    treináveis os Blocos 4 e 5 + model.classifier (requires_grad = True).
    Em CNN.features, cada bloco possui 4 camadas (Conv2d, BatchNorm2d, ReLU, MaxPool2d):
      - Blocos 1 a 3: model.features[:12] (índices 0 a 11)
      - Blocos 4 e 5: model.features[12:] (índices 12 a 19)
    """
    for param in model.features[:12].parameters():
        param.requires_grad = False

    for param in model.features[12:].parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True


def fine_tune_model():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Fine-Tuning em: {device} | Precisão Mista (AMP): {use_amp}\n")

    batch_size = BATCH_SIZE
    lr = LEARNING_RATE
    eta_min = ETA_MIN
    weight_decay = WEIGHT_DECAY
    n_epochs = N_EPOCHS
    patience = EARLY_STOPPING_PATIENCE

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pretrained_path = OUTPUT_DIR / "best_model_4.0.pth"
    save_path = OUTPUT_DIR / "best_model_4.0_Final.pth"
    history_path = OUTPUT_DIR / "training_history_4.0_Final.csv"
    val_preds_path = OUTPUT_DIR / "val_predictions_4.0_Final.csv"

    # 1. DataLoaders com Data Augmentation suave no treino
    finetune_transform = get_finetune_transform(rotation_degrees=ROTATION_DEGREES)
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        train_transform=finetune_transform,
    )

    # 2. Instanciar a arquitetura da V4 e carregar os pesos salvos
    model = CNN().to(device)
    state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Pesos carregados de: {pretrained_path}")

    # 3. Congelar do Bloco 1 ao Bloco 3 (manter Blocos 4, 5 e classificador treináveis)
    freeze_initial_blocks(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(
        f"Parâmetros treináveis: {trainable_count:,} / {total_params:,} "
        f"({100.0 * trainable_count / total_params:.2f}%)\n"
    )

    # 4. Otimizador, perda, scheduler e scaler AMP
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=eta_min
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    # 5. Loop de Épocas
    for epoch in range(n_epochs):
        # TREINO:
        model.train()
        train_loss, train_correct = 0.0, 0

        train_bar = tqdm(
            train_loader, desc=f"Fine-Tune Epoch [{epoch + 1}/{n_epochs}] Treino"
        )
        for images, labels in train_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1).float()

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs.detach()) > 0.5).float()
            train_correct += (preds == labels).sum().item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / len(train_loader.dataset)

        # VALIDAÇÃO:
        model.eval()
        val_loss, val_correct = 0.0, 0
        val_true_list, val_prob_list, val_pred_list = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).unsqueeze(1).float()

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                probs = torch.sigmoid(outputs.float())
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
            f"Epoch [{epoch + 1}/{n_epochs}] (LR: {current_lr:.8f})\n"
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
                print(
                    f"\nEarly Stopping acionado após {patience} épocas sem melhora na Val Loss."
                )
                break

    print(f"\nHistórico de fine-tuning salvo em: {history_path}")
    print(f"Previsões de validação do melhor modelo salvas em: {val_preds_path}")


if __name__ == "__main__":
    fine_tune_model()
