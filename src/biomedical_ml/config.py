"""Project-wide paths and seeding.

Every stochastic step in this project routes through :func:`set_seed` so that a
clean checkout plus ``scripts/`` reproduces the reported numbers exactly.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def set_seed(seed: int = SEED) -> int:
    """Seed Python, NumPy and (when installed) PyTorch. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def ensure_dirs() -> None:
    """Create the output directories that are gitignored or start empty."""
    for directory in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_config(path: Path = CONFIG_DIR / "default.yaml") -> dict[str, Any]:
    """Load an experiment config so reported numbers trace back to one file.

    ``configs/default.yaml`` documents the settings behind every reported
    result (split strategy, feature count, which metrics get reported); reading
    it here means code and documentation cannot silently drift apart.
    """
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
