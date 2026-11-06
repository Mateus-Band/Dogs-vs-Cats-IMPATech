"""Script para prever o resultado em novas imagens"""

import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from dataset import ROOT_DIR, get_norm_stats, get_test_dataloader
from model import CNN

OUTPUT_DIR = ROOT_DIR / "data"


def predict_single_image(image_path: str | Path, model_path: Path = OUTPUT_DIR / "best_model_3.0.pth") -> tuple[str, float]:
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mean, std = get_norm_stats()
    transform = Compose(
        [
            Resize(384),
            CenterCrop(384),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob_dog = torch.sigmoid(output).item()

    if prob_dog >= 0.5:
        predicted_class = "Cachorro"
        confidence = prob_dog * 100
    else:
        predicted_class = "Gato"
        confidence = (1 - prob_dog) * 100

    print(f"\nImagem: {image_path.name}")
    print(f"Predição: {predicted_class} ({confidence:.2f}% de confiança)")
    print(f"Probabilidade (Cachorro): {prob_dog:.4f}")

    return predicted_class, prob_dog


def generate_predictions():
    # 1. Configurações e Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Gerando previsões no dispositivo: {device}\n")

    batch_size = 48
    model_path = OUTPUT_DIR / "best_model_3.0.pth"
    output_path = OUTPUT_DIR / "submission_3.0.csv"

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
    parser = argparse.ArgumentParser(description="Gera previsões com o modelo de classificação.")
    parser.add_argument(
        "image",
        nargs="?",
        type=str,
        default=None,
        help="Caminho para uma única imagem (ex: ~/Downloads/foto.jpg). Se omitido, processa o lote de teste para submissão.",
    )
    args = parser.parse_args()

    if args.image:
        predict_single_image(args.image)
    else:
        generate_predictions()
