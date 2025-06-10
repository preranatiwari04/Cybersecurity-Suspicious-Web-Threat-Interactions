import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"

# Data URLs (add your dataset links here)
DATASET_URLS = {
    "cyber_threats": "https://your-dataset-url.com/cybersecurity_data.csv",
    "malware_samples": "https://another-dataset-url.com/malware_data.csv"
}

# Model parameters
MODEL_CONFIGS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    "logistic_regression": {
        "C": 1.0,
        "random_state": 42,
        "max_iter": 1000
    }
}

# Database settings
DATABASE_CONFIG = {
    "sqlite_path": str(DATA_DIR / "cybersecurity.db"),
    "table_names": {
        "threats": "threat_data",
        "summary": "threat_summary",
        "models": "model_performance"
    }
}

# Visualization settings
PLOT_CONFIG = {
    "style": "seaborn-v0_8",
    "figure_size": (12, 8),
    "color_palette": "husl",
    "save_format": "png",
    "dpi": 300
}