"""Tests for paths, seeding, and config loading.

Everything else in this project imports from here, so it's worth its own
direct coverage rather than relying on other tests to exercise it indirectly.
"""

from __future__ import annotations

import random
import sys
from unittest.mock import patch

import numpy as np
import yaml

from biomedical_ml.config import CONFIG_DIR, ensure_dirs, load_config, set_seed


def test_set_seed_returns_the_seed_it_was_given():
    assert set_seed(7) == 7


def test_set_seed_makes_python_random_reproducible():
    set_seed(123)
    first = [random.random() for _ in range(5)]
    set_seed(123)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_set_seed_makes_numpy_reproducible():
    set_seed(123)
    first = np.random.rand(5)
    set_seed(123)
    second = np.random.rand(5)

    np.testing.assert_array_equal(first, second)


def test_set_seed_still_returns_seed_when_torch_is_unavailable():
    # Simulate torch not being installed: sys.modules["torch"] = None makes
    # `import torch` raise ImportError, exercising the fallback return.
    with patch.dict(sys.modules, {"torch": None}):
        assert set_seed(5) == 5


def test_ensure_dirs_creates_expected_directories(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    processed = tmp_path / "data" / "processed"
    results = tmp_path / "results"
    figures = tmp_path / "figures"

    monkeypatch.setattr("biomedical_ml.config.RAW_DIR", raw)
    monkeypatch.setattr("biomedical_ml.config.PROCESSED_DIR", processed)
    monkeypatch.setattr("biomedical_ml.config.RESULTS_DIR", results)
    monkeypatch.setattr("biomedical_ml.config.FIGURES_DIR", figures)

    for directory in (raw, processed, results, figures):
        assert not directory.exists()

    ensure_dirs()

    for directory in (raw, processed, results, figures):
        assert directory.is_dir()


def test_load_config_reads_the_default_yaml():
    cfg = load_config()

    assert cfg["seed"] == 42
    for section in ("data", "split", "features", "metrics", "autoencoder"):
        assert section in cfg


def test_load_config_reads_an_arbitrary_path(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.dump({"seed": 1, "note": "test"}), encoding="utf-8")

    cfg = load_config(custom)

    assert cfg == {"seed": 1, "note": "test"}


def test_default_yaml_lives_where_config_dir_points():
    assert (CONFIG_DIR / "default.yaml").is_file()
