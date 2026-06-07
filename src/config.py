# src/config.py

from pathlib import Path

# Project root (src/config.py -> src -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "powerpredict.csv.zip"

# Models
MODEL_DIR = PROJECT_ROOT / "models"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2