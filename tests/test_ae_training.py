"""Tests for autoencoder training: early stopping, the train/val split, and checkpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from biomedical_ml.ae_training import (
    EarlyStopping,
    build_checkpoint,
    build_model_from_checkpoint,
    encode_all,
    load_checkpoint,
    prepare_ae_data,
    save_checkpoint,
    train_autoencoder,
    transform_with_checkpoint,
)
from biomedical_ml.preprocessing import Dataset
from biomedical_ml.splits import assert_no_subject_leakage


class TestEarlyStopping:
    def test_stops_after_patience_bad_epochs(self):
        stopper = EarlyStopping(patience=2)

        results = [stopper.step(v, epoch) for epoch, v in enumerate([1.0, 0.9, 0.95, 0.96])]

        assert results == [False, False, False, True]

    def test_tracks_best_value_and_epoch(self):
        stopper = EarlyStopping(patience=5)

        for epoch, v in enumerate([1.0, 0.9, 0.95, 0.8, 0.85]):
            stopper.step(v, epoch)

        assert stopper.best == pytest.approx(0.8)
        assert stopper.best_epoch == 3

    def test_min_delta_requires_meaningful_improvement(self):
        # A tiny improvement smaller than min_delta must not reset the counter.
        stopper = EarlyStopping(patience=2, min_delta=0.05)

        results = [stopper.step(v, epoch) for epoch, v in enumerate([1.0, 0.98, 0.97])]

        assert results == [False, False, True]

    def test_never_stops_while_still_improving(self):
        stopper = EarlyStopping(patience=1)

        results = [stopper.step(v, epoch) for epoch, v in enumerate([1.0, 0.5, 0.1, 0.01])]

        assert results == [False, False, False, False]


@pytest.fixture
def synthetic_dataset() -> Dataset:
    """A small cohort shaped like GSE138458: repeat visits, imbalanced, with signal.

    Sized like test_evaluation.py's cohort_with_signal fixture — enough subjects
    per class for StratifiedGroupKFold to form a sane 80/20 split.
    """
    rng = np.random.default_rng(0)
    subjects, labels = [], []
    for subject in range(24):
        visits = 2 if subject % 2 == 0 else 1
        subjects += [f"case{subject}"] * visits
        labels += [1] * visits
    for subject in range(8):
        visits = 2 if subject % 4 == 0 else 1
        subjects += [f"ctrl{subject}"] * visits
        labels += [0] * visits

    order = rng.permutation(len(subjects))
    samples = [f"GSM{i}" for i in range(len(subjects))]
    groups = pd.Series([subjects[i] for i in order], index=samples, name="subject_id")
    y = pd.Series([labels[i] for i in order], index=samples, name="sle")

    n = len(y)
    informative = y.to_numpy() * 2.0 + rng.normal(scale=1.0, size=n)
    noise = rng.normal(size=(n, 9))
    X = pd.DataFrame(
        np.column_stack([informative, noise]).astype("float32"),
        index=pd.Index(samples, name="geo_accession"),
        columns=["informative", *[f"noise{i}" for i in range(9)]],
    )
    metadata = pd.DataFrame(index=X.index)

    return Dataset(X=X, y=y, groups=groups, metadata=metadata)


def test_prepare_ae_data_has_no_subject_leakage(synthetic_dataset):
    _, _, _, train_idx, val_idx = prepare_ae_data(synthetic_dataset, k=5, n_splits=5, seed=0)

    assert_no_subject_leakage(synthetic_dataset.groups, train_idx, val_idx)


def test_prepare_ae_data_selects_exactly_k_features(synthetic_dataset):
    X_train, X_val, _, _, _ = prepare_ae_data(synthetic_dataset, k=5, n_splits=5, seed=0)

    assert X_train.shape[1] == 5
    assert X_val.shape[1] == 5


def test_train_autoencoder_produces_matching_history_lengths(synthetic_dataset):
    result = train_autoencoder(
        synthetic_dataset, k=5, latent_dim=2, hidden_dims=(4,), max_epochs=5, patience=10, seed=0
    )

    assert len(result.history["train_loss"]) == len(result.history["val_loss"])
    assert 1 <= len(result.history["train_loss"]) <= 5


def test_train_autoencoder_reports_a_valid_best_epoch(synthetic_dataset):
    result = train_autoencoder(
        synthetic_dataset, k=5, latent_dim=2, hidden_dims=(4,), max_epochs=5, patience=10, seed=0
    )

    assert 0 <= result.best_epoch < len(result.history["val_loss"])
    assert result.best_val_loss == pytest.approx(min(result.history["val_loss"]))


def test_train_autoencoder_has_no_subject_leakage(synthetic_dataset):
    result = train_autoencoder(
        synthetic_dataset, k=5, latent_dim=2, hidden_dims=(4,), max_epochs=3, patience=10, seed=0
    )

    assert_no_subject_leakage(synthetic_dataset.groups, result.train_idx, result.val_idx)


def test_train_autoencoder_model_matches_configured_latent_dim(synthetic_dataset):
    result = train_autoencoder(
        synthetic_dataset, k=5, latent_dim=3, hidden_dims=(4,), max_epochs=3, patience=10, seed=0
    )

    assert result.config.latent_dim == 3
    assert result.model.encode(result.model.encoder[0].weight.new_zeros(1, 5)).shape == (1, 3)


def test_train_autoencoder_early_stopping_actually_stops_training(synthetic_dataset):
    # patience=1 with a deliberately large min_delta means the very first
    # non-improving epoch ends training well before max_epochs -- exercises
    # the early-exit path, not just the metric bookkeeping (already covered
    # by the EarlyStopping unit tests above).
    result = train_autoencoder(
        synthetic_dataset,
        k=5,
        latent_dim=2,
        hidden_dims=(4,),
        max_epochs=200,
        patience=1,
        min_delta=0.01,
        seed=0,
    )

    assert len(result.history["train_loss"]) < 200


def test_checkpoint_roundtrip_preserves_weights(tmp_path, synthetic_dataset):
    result = train_autoencoder(
        synthetic_dataset, k=5, latent_dim=2, hidden_dims=(4,), max_epochs=3, patience=10, seed=0
    )
    path = tmp_path / "ae_checkpoint.pt"

    save_checkpoint(result, synthetic_dataset, path)
    checkpoint = load_checkpoint(path)
    reloaded = build_model_from_checkpoint(checkpoint)

    for p1, p2 in zip(result.model.parameters(), reloaded.parameters(), strict=True):
        np.testing.assert_array_equal(p1.detach().numpy(), p2.detach().numpy())


def test_checkpoint_records_selected_probes_and_scaler(synthetic_dataset):
    result = train_autoencoder(
        synthetic_dataset, k=5, latent_dim=2, hidden_dims=(4,), max_epochs=3, patience=10, seed=0
    )

    checkpoint = build_checkpoint(result, synthetic_dataset)

    assert len(checkpoint["selected_probe_ids"]) == 5
    assert len(checkpoint["scaler_mean"]) == 5
    assert len(checkpoint["scaler_scale"]) == 5
    assert set(checkpoint["train_subject_ids"]) & set(checkpoint["val_subject_ids"]) == set()


def test_transform_with_checkpoint_matches_the_fitted_pipeline(synthetic_dataset):
    result = train_autoencoder(
        synthetic_dataset, k=5, latent_dim=2, hidden_dims=(4,), max_epochs=3, patience=10, seed=0
    )
    checkpoint = build_checkpoint(result, synthetic_dataset)

    expected = result.feature_pipeline.transform(synthetic_dataset.X).astype("float32")
    actual = transform_with_checkpoint(synthetic_dataset.X, checkpoint)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_encode_all_returns_one_row_per_input_sample_with_latent_dim_columns(synthetic_dataset):
    result = train_autoencoder(
        synthetic_dataset, k=5, latent_dim=3, hidden_dims=(4,), max_epochs=3, patience=10, seed=0
    )
    checkpoint = build_checkpoint(result, synthetic_dataset)

    latents = encode_all(result.model, synthetic_dataset.X, checkpoint)

    assert list(latents.index) == list(synthetic_dataset.X.index)
    assert list(latents.columns) == ["z0", "z1", "z2"]
    assert not latents.isna().to_numpy().any()


def test_encode_all_matches_manual_transform_then_encode(synthetic_dataset):
    result = train_autoencoder(
        synthetic_dataset, k=5, latent_dim=2, hidden_dims=(4,), max_epochs=3, patience=10, seed=0
    )
    checkpoint = build_checkpoint(result, synthetic_dataset)

    latents = encode_all(result.model, synthetic_dataset.X, checkpoint)

    features = transform_with_checkpoint(synthetic_dataset.X, checkpoint)
    with torch.no_grad():
        expected = result.model.encode(torch.from_numpy(features)).numpy()

    np.testing.assert_array_equal(latents.to_numpy(), expected)


def test_encode_all_handles_samples_never_seen_during_training(synthetic_dataset):
    # The whole point: encode the full 330-sample dataset, not just the 264
    # training rows the model was fit on.
    result = train_autoencoder(
        synthetic_dataset, k=5, latent_dim=2, hidden_dims=(4,), max_epochs=3, patience=10, seed=0
    )
    checkpoint = build_checkpoint(result, synthetic_dataset)

    latents = encode_all(result.model, synthetic_dataset.X, checkpoint)

    assert len(latents) == len(synthetic_dataset.X)
