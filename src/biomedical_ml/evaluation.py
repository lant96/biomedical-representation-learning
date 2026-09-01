"""Repeated, subject-grouped CV evaluation of classical baselines.

Every metric here is computed per fold and aggregated afterwards — never on a
single split, and never plain accuracy, which a model could win at a 13:1
class ratio just by predicting SLE every time. See
:mod:`biomedical_ml.splits` for why the folds are grouped and repeated.

F1 and balanced accuracy also need a decision threshold, and a blind 0.5 cutoff
would understate a model at this imbalance; :func:`_best_threshold` tunes it
per fold on the training data alone (see its docstring).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from biomedical_ml.config import SEED
from biomedical_ml.models import build_pipeline
from biomedical_ml.splits import repeated_cv_splits

METRICS = ("roc_auc", "pr_auc", "f1", "balanced_accuracy")


def _set_xgboost_class_balance(pipeline, y_train: np.ndarray) -> None:
    """Set XGBoost's ``scale_pos_weight`` from the training fold's own label ratio.

    XGBoost has no ``class_weight='balanced'`` option the way sklearn models
    do; ``scale_pos_weight`` is its equivalent. Computing it from ``y_train``
    only (never the test fold) keeps this leak-free — it is just a label
    count, refit per fold like everything else in the pipeline.
    """
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "scale_pos_weight"):
        return
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    pipeline.set_params(clf__scale_pos_weight=n_neg / max(n_pos, 1))


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """The decision threshold that maximises balanced accuracy on ``(y_true, proba)``.

    At a 13:1 class ratio, a blind 0.5 cutoff on ``predict()`` tends to
    under-predict the minority class; F1 and balanced accuracy computed at
    0.5 would understate what the model's ranking actually supports. Scanning
    the observed probabilities (as an ROC curve does) is exact and cheap at
    this sample size.
    """
    candidates = np.concatenate(([0.0], np.unique(proba), [1.0]))
    scores = [balanced_accuracy_score(y_true, proba >= t) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def _fit_and_score(
    pipeline, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    _set_xgboost_class_balance(pipeline, y_train)
    pipeline.fit(X_train, y_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]

    # Threshold tuned on the training fold's own (in-sample) predictions only,
    # never on the test fold, so F1/balanced accuracy reflect a realistic
    # operating point without leaking test-fold information into it.
    proba_train = pipeline.predict_proba(X_train)[:, 1]
    threshold = _best_threshold(y_train, proba_train)
    pred_test = (proba_test >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, proba_test),
        "pr_auc": average_precision_score(y_test, proba_test),
        "f1": f1_score(y_test, pred_test, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
        "threshold": threshold,
    }


def evaluate_pipeline_factory(
    name: str,
    pipeline_factory: Callable[[int], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Repeated grouped-stratified CV for an arbitrary pipeline. One row per fold.

    The primitive :func:`evaluate_model` is built on. Exists on its own for
    callers whose pipeline isn't one of the registered classical baselines —
    Day 4's linear probe on frozen autoencoder embeddings, in particular,
    needs the same repeated grouped CV and threshold-tuning machinery but a
    different, already-dense feature space with no ``SelectKBest`` step. A
    fresh pipeline is built for every fold (rather than cloning one), so
    anything the factory fits is always refit on that fold's training data
    alone.
    """
    X_values, y_values = X.to_numpy(), y.to_numpy()
    rows = []

    splits = repeated_cv_splits(y, groups, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    for i, (train_idx, test_idx) in enumerate(splits):
        repeat, fold = divmod(i, n_splits)
        pipeline = pipeline_factory(seed)
        scores = _fit_and_score(
            pipeline,
            X_values[train_idx],
            y_values[train_idx],
            X_values[test_idx],
            y_values[test_idx],
        )
        rows.append({"model": name, "repeat": repeat, "fold": fold, **scores})

    return pd.DataFrame(rows)


def evaluate_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    k: int = 2000,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Score one registered classical baseline (see :mod:`biomedical_ml.models`)."""
    return evaluate_pipeline_factory(
        model_name,
        lambda s: build_pipeline(model_name, k=k, seed=s),
        X,
        y,
        groups,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
    )


def evaluate_models(
    model_names: Sequence[str],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    k: int = 2000,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Run :func:`evaluate_model` for each name and stack the results."""
    return pd.concat(
        [
            evaluate_model(
                name, X, y, groups, k=k, n_splits=n_splits, n_repeats=n_repeats, seed=seed
            )
            for name in model_names
        ],
        ignore_index=True,
    )


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Mean/std per model across all folds, ranked by mean ROC-AUC.

    ROC-AUC is the primary metric (see ``configs/default.yaml``); the others
    are reported alongside it rather than used to break ties, since with 22
    control subjects the ranking below the top model is not reliable.
    """
    summary = results.groupby("model")[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.sort_values("roc_auc_mean", ascending=False)


def best_model(results: pd.DataFrame) -> str:
    """The model name with the highest mean ROC-AUC."""
    return summarize_results(results).index[0]
