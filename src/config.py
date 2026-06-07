# src/config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "powerpredict.csv.zip"

MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42

TEST_SIZE = 0.2