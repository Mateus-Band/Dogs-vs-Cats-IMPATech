"""Script para prever o resultado em novas imagens"""

import pandas as pd
import torch
from tqdm import tqdm

from dataset import ROOT_DIR, get_test_dataloader
from model import CNN

OUTPUT_DIR = ROOT_DIR / "data"


def generate_predictions():
    # 1. Configurações e Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Gerando previsões no dispositivo: {device}\n")

    batch_size = 64
    model_path = OUTPUT_DIR / "best_model.pth"
    output_path = OUTPUT_DIR / "submission.csv"

    # 2. Carregar Dataloader de Teste e Modelo
    test_loader = get_test_dataloader(batch_size=batch_size)

    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Modo de avaliação (desativa Dropout e fixa BatchNorm)

    ids = []
    probabilities = []

    # 3. Loop de Inferência
    with torch.no_grad():  # Desativa gradientes para poupar memória e tempo
        test_bar = tqdm(test_loader, desc="Processando imagens de teste")
        for images, image_ids in test_bar:
            images = images.to(device)

            # Passa as imagens pela rede e converte os logits em probabilidades [0.0, 1.0]
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            # Trata casos em que o lote final possa ter tamanho 1
            if probs.ndim == 0:
                probs = [float(probs)]
            else:
                probs = probs.tolist()

            ids.extend(image_ids.numpy() if isinstance(image_ids, torch.Tensor) else image_ids)
            probabilities.extend(probs)

    # 4. Criar DataFrame e Exportar CSV
    df = pd.DataFrame(
        {
            "id": ids,
            "label": probabilities,
        }
    )

    # Garante ordenação sequencial pelos IDs das imagens
    df = df.sort_values("id").reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"\nPrevisões salvas com sucesso em '{output_path}'!")
    print("\nPrimeiras linhas do arquivo gerado:")
    print(df.head())


if __name__ == "__main__":
    generate_predictions()
