"""Train the autoencoder with a subject-grouped train/val split and early stopping.

Reuses :mod:`biomedical_ml.splits` for the split (grouped on ``subject_id``, so
no patient's repeat visit spans train and val) and
:func:`biomedical_ml.models.build_feature_pipeline` for the input space (fit on
the training rows only, so val never leaks into feature selection or scaling).
The label is used only to build that split and to run ``SelectKBest``'s
ANOVA F-test — the training loss itself never sees it.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from biomedical_ml.autoencoder import Autoencoder, AutoencoderConfig
from biomedical_ml.config import SEED, set_seed
from biomedical_ml.models import build_feature_pipeline
from biomedical_ml.preprocessing import Dataset
from biomedical_ml.splits import holdout_split


class EarlyStopping:
    """Stops training once validation loss goes ``patience`` checks without improving.

    Kept separate from the training loop so the stopping *decision* can be unit
    tested against a plain sequence of loss values, independent of anything
    that makes real gradient descent slow or stochastic.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = -1
        self.num_bad_epochs = 0

    def step(self, value: float, epoch: int) -> bool:
        """Record ``value`` for ``epoch``; return whether training should stop now."""
        if value < self.best - self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


@dataclass
class AutoencoderTrainingResult:
    """Everything a caller needs after training: the model, how it got there, and the split."""

    model: Autoencoder
    config: AutoencoderConfig
    history: dict[str, list[float]]
    best_epoch: int
    best_val_loss: float
    feature_pipeline: Pipeline
    train_idx: np.ndarray
    val_idx: np.ndarray


def prepare_ae_data(
    dataset: Dataset, *, k: int, n_splits: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, Pipeline, np.ndarray, np.ndarray]:
    """Grouped-stratified train/val split, then fit-on-train feature selection + scaling.

    The split is stratified on the label purely so validation reconstruction
    loss is measured on a representative mix of both classes — the label plays
    no further role, since the autoencoder's loss never uses it.
    """
    train_idx, val_idx = holdout_split(dataset.y, dataset.groups, n_splits=n_splits, seed=seed)

    X_train_raw = dataset.X.iloc[train_idx]
    X_val_raw = dataset.X.iloc[val_idx]
    y_train = dataset.y.iloc[train_idx]

    feature_pipeline = build_feature_pipeline(k=k)
    feature_pipeline.fit(X_train_raw, y_train)

    X_train = feature_pipeline.transform(X_train_raw).astype(np.float32)
    X_val = feature_pipeline.transform(X_val_raw).astype(np.float32)

    return torch.from_numpy(X_train), torch.from_numpy(X_val), feature_pipeline, train_idx, val_idx


def train_autoencoder(
    dataset: Dataset,
    *,
    k: int = 2000,
    latent_dim: int = 32,
    hidden_dims: tuple[int, ...] = (256, 64),
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 300,
    patience: int = 20,
    min_delta: float = 1e-4,
    n_splits: int = 5,
    seed: int = SEED,
) -> AutoencoderTrainingResult:
    """Train the autoencoder with early stopping on validation reconstruction loss.

    Regularization is dropout (in the architecture) plus L2 weight decay (in
    the optimizer); early stopping restores the weights from the
    best-validation-loss epoch, not whichever epoch training happened to end
    on.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_val, feature_pipeline, train_idx, val_idx = prepare_ae_data(
        dataset, k=k, n_splits=n_splits, seed=seed
    )
    input_dim = X_train.shape[1]

    config = AutoencoderConfig(
        input_dim=input_dim, latent_dim=latent_dim, hidden_dims=tuple(hidden_dims), dropout=dropout
    )
    model = Autoencoder(config).to(device)

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    X_val_device = X_val.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
            n_seen += batch.size(0)
        train_loss = running_loss / n_seen

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_device), X_val_device).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        should_stop = stopper.step(val_loss, epoch)
        if stopper.best_epoch == epoch:
            best_state = copy.deepcopy(model.state_dict())
        if should_stop:
            break

    model.load_state_dict(best_state)
    model.eval()

    return AutoencoderTrainingResult(
        model=model,
        config=config,
        history=history,
        best_epoch=stopper.best_epoch,
        best_val_loss=stopper.best,
        feature_pipeline=feature_pipeline,
        train_idx=train_idx,
        val_idx=val_idx,
    )


def build_checkpoint(result: AutoencoderTrainingResult, dataset: Dataset) -> dict[str, Any]:
    """Assemble a plain-data checkpoint dict, ready for ``torch.save``.

    Feature selection and scaling are stored as raw arrays (selected probe IDs,
    scaler mean/scale) rather than the fitted sklearn ``Pipeline`` itself, so
    reloading a checkpoint never depends on unpickling a matching scikit-learn
    version — :func:`transform_with_checkpoint` reapplies them with plain numpy.
    """
    selector = result.feature_pipeline.named_steps["select"]
    scaler = result.feature_pipeline.named_steps["scale"]
    selected_probe_ids = dataset.X.columns[selector.get_support()].tolist()

    return {
        "model_state_dict": result.model.state_dict(),
        "autoencoder_config": asdict(result.config),
        "selected_probe_ids": selected_probe_ids,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "history": result.history,
        "train_subject_ids": sorted(set(dataset.groups.iloc[result.train_idx])),
        "val_subject_ids": sorted(set(dataset.groups.iloc[result.val_idx])),
        "seed": SEED,
    }


def save_checkpoint(result: AutoencoderTrainingResult, dataset: Dataset, path: Path) -> None:
    """Build and write a checkpoint in one step (see :func:`build_checkpoint`)."""
    torch.save(build_checkpoint(result, dataset), path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    ``weights_only=False`` is safe here because the checkpoint holds only
    plain Python data and tensors we wrote ourselves — never load a checkpoint
    from an untrusted source this way.
    """
    return torch.load(path, weights_only=False)


def build_model_from_checkpoint(checkpoint: dict[str, Any]) -> Autoencoder:
    """Reconstruct a ready-to-use (``eval`` mode) model from a checkpoint dict."""
    config = AutoencoderConfig(**checkpoint["autoencoder_config"])
    model = Autoencoder(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def transform_with_checkpoint(X: pd.DataFrame, checkpoint: dict[str, Any]) -> np.ndarray:
    """Apply a checkpoint's exact probe selection and standardization to new data.

    Lets Day 4 (or anything else) reproduce the model's input space for
    samples the autoencoder never trained or validated on, without needing the
    original fitted ``SelectKBest``/``StandardScaler`` objects.
    """
    selected = checkpoint["selected_probe_ids"]
    mean = np.asarray(checkpoint["scaler_mean"])
    scale = np.asarray(checkpoint["scaler_scale"])
    return ((X[selected].to_numpy() - mean) / scale).astype(np.float32)


def encode_all(model: Autoencoder, X: pd.DataFrame, checkpoint: dict[str, Any]) -> pd.DataFrame:
    """Encode every row of ``X`` through a frozen, checkpointed autoencoder.

    Applies the checkpoint's exact feature selection and standardization
    first, so ``X`` can be the full, unselected probe matrix — including rows
    the model never saw during training or validation (Day 4 extracts
    embeddings for all 330 samples this way, not just the 264 training ones).
    """
    model.eval()
    features = transform_with_checkpoint(X, checkpoint)
    with torch.no_grad():
        latent = model.encode(torch.from_numpy(features)).numpy()
    columns = [f"z{i}" for i in range(latent.shape[1])]
    return pd.DataFrame(latent, index=X.index, columns=columns)
