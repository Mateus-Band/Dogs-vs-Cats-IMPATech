"""Leitura e preprocessamento dos dados"""

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

# Caminhos base resolvidos a partir da localização deste arquivo
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "dogs-vs-cats"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test1"


def get_default_transform():
    return Compose(
        [
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    all_files = sorted(train_dir.glob("*.jpg"))
    if max_samples is not None:
        # Pega metade do início (cats) e metade do fim (dogs) para manter ambas as classes ao usar stratify
        half = max_samples // 2
        all_files = all_files[:half] + all_files[-half:]

    # cat = 0, dog = 1
    labels = [0 if f.name.startswith("cat.") else 1 for f in all_files]

    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    transform = get_default_transform()

    train_dataset = DogsCatsDataset(train_files, train_labels, transform=transform)
    val_dataset = DogsCatsDataset(val_files, val_labels, transform=transform)

    return train_dataset, val_dataset


def get_dataloaders(
    batch_size: int = 64,
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
    batch_size: int = 64,
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
