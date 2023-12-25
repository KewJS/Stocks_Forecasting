import os
from pathlib import Path

PARENT_DIR = Path(__file__).parent.resolve().parent.parent
DATA_DIR = os.path.join(PARENT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
PREPROCESS_DATA_DIR = os.path.join(DATA_DIR, "preprocess_data")
MODELS_DIR = os.path.join(DATA_DIR, "models")

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(MODELS_DIR).exists():
    os.mkdir(MODELS_DIR)