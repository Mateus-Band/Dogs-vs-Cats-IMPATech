"""Leitura e preprocessamento dos dados"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    ToTensor,
)

# Caminhos base resolvidos a partir da localização deste arquivo
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "dogs-vs-cats"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test1"


NORM_STATS_PATH = ROOT_DIR / "data" / "norm_stats.json"


def compute_channel_stats(file_paths: list[Path]) -> tuple[list[float], list[float]]:
    """Média e desvio padrão por canal (R, G, B) dos pixels em escala [0, 1]."""
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


def get_train_files(train_dir: Path = TRAIN_DIR, max_samples: int | None = None):
    all_files = sorted(train_dir.glob("*.jpg"))
    if max_samples is not None:
        # Pega metade do início (cats) e metade do fim (dogs) para manter ambas as classes ao usar stratify
        half = max_samples // 2
        all_files = all_files[:half] + all_files[-half:]

    # cat = 0, dog = 1
    labels = [0 if f.name.startswith("cat.") else 1 for f in all_files]

    return train_test_split(
        all_files,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )


def get_norm_stats() -> tuple[list[float], list[float]]:
    """Carrega as estatísticas do split de treino completo (calcula e salva se não existirem)."""
    if NORM_STATS_PATH.exists():
        stats = json.loads(NORM_STATS_PATH.read_text())
        return stats["mean"], stats["std"]

    train_files, _, _, _ = get_train_files()
    mean, std = compute_channel_stats(train_files)
    NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NORM_STATS_PATH.write_text(json.dumps({"mean": mean, "std": std}, indent=2))
    print(f"Estatísticas salvas em {NORM_STATS_PATH}: mean={mean}, std={std}")
    return mean, std


def get_train_transform(mean: list[float] | None = None, std: list[float] | None = None):
    if mean is None or std is None:
        mean, std = get_norm_stats()
    return Compose(
        [
            RandomResizedCrop(224, scale=(0.8, 1.0)),
            RandomHorizontalFlip(),
            RandomRotation(15),
            ColorJitter(brightness=0.2, contrast=0.2),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


def get_default_transform(mean: list[float] | None = None, std: list[float] | None = None):
    if mean is None or std is None:
        mean, std = get_norm_stats()
    return Compose(
        [
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


class DogsCatsDataset(Dataset):
    def __init__(self, file_paths: list[Path], labels: list[int] | None = None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label

        image_id = int(img_path.stem)
        return image, image_id


def get_train_val_datasets(train_dir: Path = TRAIN_DIR, max_samples: int | None = None):
    train_files, val_files, train_labels, val_labels = get_train_files(train_dir, max_samples)

    if max_samples is None:
        train_transform = get_train_transform()
        val_transform = get_default_transform()
    else:
        # Teste leve: estatísticas do subconjunto de treino (sem sobrescrever o arquivo salvo)
        mean, std = compute_channel_stats(train_files)
        train_transform = get_train_transform(mean, std)
        val_transform = get_default_transform(mean, std)

    train_dataset = DogsCatsDataset(train_files, train_labels, transform=train_transform)
    val_dataset = DogsCatsDataset(val_files, val_labels, transform=val_transform)

    return train_dataset, val_dataset


def get_dataloaders(
    batch_size: int = 128,
    train_dir: Path = TRAIN_DIR,
    max_samples: int | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    train_dataset, val_dataset = get_train_val_datasets(train_dir, max_samples=max_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def get_test_dataloader(
    batch_size: int = 128,
    test_dir: Path = TEST_DIR,
    max_samples: int | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    # Ordena numericamente pelo ID do arquivo (1.jpg, 2.jpg, ...)
    test_files = sorted(test_dir.glob("*.jpg"), key=lambda f: int(f.stem))
    if max_samples is not None:
        test_files = test_files[:max_samples]

    transform = get_default_transform()

    test_dataset = DogsCatsDataset(test_files, labels=None, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader
