import torch
import pathlib
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    DIR_PATH: Path = pathlib.Path.cwd()
    CELEBA_DIR: Path = Path(DIR_PATH, 'celeba')
    IMG_DIR_TRAIN: Path = Path(CELEBA_DIR, 'train')
    IMG_DIR_VAL: Path = Path(CELEBA_DIR, 'test')
    CSV_TRAIN: Path = Path(CELEBA_DIR, 'train.csv')
    CSV_VAL: Path = Path(CELEBA_DIR, 'test.csv')
    TEST_DIR: Path = Path(DIR_PATH, 'test_images')
    DEVICE: str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    MODEL_PATH = Path(DIR_PATH, 'logdir', 'checkpoints')
    LR: float = 0.0001
    BATCH_SIZE: int = 64
    EPOCH: int = 20
    STEPS_PER_EPOCH: int = 51




