"""Subject-aware splitting for GSE138458.

GSE138458 has 330 samples drawn from 218 subjects — 102 subjects contributed two
visits and 5 contributed three. A plain ``StratifiedKFold`` over samples would
routinely place one visit from a patient in train and another in test. Because
repeat visits from the same patient share that patient's baseline expression,
the model can recognise the *individual* rather than the disease, and the
reported AUC is optimistically biased.

Every splitter here therefore groups on ``subject_id`` while still stratifying
on the label, which matters just as much given 23 controls against 307 cases.

The binding constraint on this dataset is the 22 control *subjects*: with five
folds a fold holds only four or five of them, so single-split estimates are
noisy. Prefer :func:`repeated_cv_splits` for headline numbers.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from biomedical_ml.config import SEED


def cv_splitter(n_splits: int = 5, seed: int = SEED) -> StratifiedGroupKFold:
    """Stratified, subject-grouped k-fold — the default CV for this project."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def holdout_split(
    y: pd.Series, groups: pd.Series, *, n_splits: int = 5, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Carve off one stratified, subject-grouped fold as a held-out test set.

    Taking a single fold of a ``StratifiedGroupKFold`` (rather than using
    ``train_test_split``) is what buys both properties at once: no subject spans
    the boundary, and the case/control ratio is preserved on each side.

    Returns:
        ``(train_idx, test_idx)`` as positional indices into ``y``.
    """
    splitter = cv_splitter(n_splits=n_splits, seed=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups))
    return train_idx, test_idx


def repeated_cv_splits(
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = SEED,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield splits from several re-seeded runs of grouped stratified k-fold.

    sklearn has no ``RepeatedStratifiedGroupKFold``, and with only 22 control
    subjects a single 5-fold run is too noisy to compare models on, so we repeat
    the whole k-fold with a different shuffle each time.
    """
    placeholder = np.zeros(len(y))
    for repeat in range(n_repeats):
        splitter = cv_splitter(n_splits=n_splits, seed=seed + repeat)
        yield from splitter.split(placeholder, y, groups)


def assert_no_subject_leakage(
    groups: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    """Raise if any subject appears on both sides of a split."""
    shared = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    if shared:
        raise AssertionError(
            f"{len(shared)} subject(s) appear in both train and test: {sorted(shared)[:5]}"
        )


def describe_split(
    y: pd.Series, groups: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray
) -> pd.DataFrame:
    """Tabulate samples, subjects and class balance on each side of a split."""
    rows = {}
    for name, idx in (("train", train_idx), ("test", test_idx)):
        y_part, g_part = y.iloc[idx], groups.iloc[idx]
        rows[name] = {
            "samples": len(idx),
            "subjects": g_part.nunique(),
            "sle_samples": int((y_part == 1).sum()),
            "control_samples": int((y_part == 0).sum()),
            "control_subjects": g_part[y_part == 0].nunique(),
        }
    return pd.DataFrame(rows).T