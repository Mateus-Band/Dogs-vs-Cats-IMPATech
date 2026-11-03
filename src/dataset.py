"""Leitura e preprocessamento dos dados"""

from pathlib import Path

# Caminhos base resolvidos a partir da localização deste arquivo
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "dogs-vs-cats"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test1"

from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
import PIL
from sklearn.model_selection import train_test_split

# separar 80% treino e 20% validação dentre o TRAIN_DIR

