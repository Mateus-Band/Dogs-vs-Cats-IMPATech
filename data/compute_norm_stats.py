"""Script para calcular a média e desvio padrão por canal (R, G, B)
das imagens de treino (após redimensionamento) e salvar em norm_stats.json.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent
TRAIN_DIR = DATA_DIR / "dogs-vs-cats" / "train"
OUTPUT_PATH = DATA_DIR / "norm_stats_3.0.json"


def get_train_files(train_dir: Path) -> list[Path]:
    """Obtém os arquivos de treino usando o mesmo split estratificado de 80% do dataset."""
    all_files = sorted(train_dir.glob("*.jpg"))
    labels = [0 if f.name.startswith("cat.") else 1 for f in all_files]

    train_files, _, _, _ = train_test_split(
        all_files,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    return train_files


def compute_channel_stats(file_paths: list[Path]) -> tuple[list[float], list[float]]:
    """Calcula média e desvio padrão por canal (R, G, B) dos pixels em escala [0, 1]."""
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for f in tqdm(file_paths, desc="Calculando média/desvio por canal (treino)"):
        with Image.open(f) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        pixels = arr.reshape(-1, 3)
        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += (pixels**2).sum(axis=0)
        n_pixels += pixels.shape[0]

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sq_sum / n_pixels - mean**2)
    return mean.tolist(), std.tolist()


def main():
    print(f"Buscando imagens de treino em: {TRAIN_DIR}")
    train_files = get_train_files(TRAIN_DIR)
    print(f"Total de imagens no split de treino: {len(train_files)}")

    mean, std = compute_channel_stats(train_files)

    stats = {
        "mean": mean,
        "std": std,
    }

    OUTPUT_PATH.write_text(json.dumps(stats, indent=2))
    print(f"\nEstatísticas salvas com sucesso em: {OUTPUT_PATH}")
    print(f"Média: {mean}")
    print(f"Desvio Padrão: {std}")


if __name__ == "__main__":
    main()

