"""Classical baseline models for GSE138458.

Every model shares one pipeline shape — ``SelectKBest`` then ``StandardScaler``
then classifier — so that feature selection is refit inside each CV fold
(never on the full matrix) and every model sees the same feature set. Scaling
is a no-op for the tree models but harmless, and keeping it uniform means one
code path handles all five baselines.

Logistic regression uses the current scikit-learn API: ``penalty`` is
deprecated in favour of ``l1_ratio`` directly on ``LogisticRegression``
(``l1_ratio=0`` is L2, ``l1_ratio=1`` is L1, anything in between is elastic
net). ``saga`` is the solver that supports all three; plain L2 also works with
the faster ``lbfgs`` default.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from biomedical_ml.config import SEED


def _logistic_regression(l1_ratio: float, seed: int) -> LogisticRegression:
    solver = "lbfgs" if l1_ratio == 0.0 else "saga"
    return LogisticRegression(
        l1_ratio=l1_ratio,
        solver=solver,
        C=1.0,
        # saga on 2000 standardized features needs ~400-2600 iterations to reach
        # tol=1e-4 (30+s per fold); relaxing to tol=1e-3 converges cleanly
        # (no ConvergenceWarning) in ~1000 iterations at ~5x the speed, with
        # identical held-out ranking observed on this dataset.
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=seed,
    )


#: Model factories, keyed by name. Each takes a seed and returns a fresh,
#: unfitted estimator — kept as factories rather than instances so every CV
#: fold gets its own independent estimator.
MODEL_FACTORIES: dict[str, Callable[[int], BaseEstimator]] = {
    "logreg_l2": lambda seed: _logistic_regression(0.0, seed),
    "logreg_l1": lambda seed: _logistic_regression(1.0, seed),
    "logreg_elasticnet": lambda seed: _logistic_regression(0.5, seed),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ),
    "xgboost": lambda seed: XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    ),
}

MODEL_NAMES: tuple[str, ...] = tuple(MODEL_FACTORIES)


def build_feature_pipeline(*, k: int = 2000) -> Pipeline:
    """The ``SelectKBest -> StandardScaler`` steps every classifier pipeline uses.

    Split out on its own so a caller that isn't a classifier — the Day 3
    autoencoder, in particular — can put its input on exactly the same
    selected-and-scaled feature space as the classical baselines, making the
    two comparable, without dragging a classifier step along with it.
    """
    return Pipeline(
        [
            ("select", SelectKBest(f_classif, k=k)),
            ("scale", StandardScaler()),
        ]
    )


def build_pipeline(model_name: str, *, k: int = 2000, seed: int = SEED) -> Pipeline:
    """Build a fresh ``SelectKBest -> StandardScaler -> classifier`` pipeline.

    Args:
        model_name: One of :data:`MODEL_NAMES`.
        k: Probes to keep. ``SelectKBest`` is refit on whatever data ``.fit()``
            is later called with, so building the pipeline here does not
            itself touch any data.
        seed: Passed through to the classifier for reproducibility.
    """
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model {model_name!r}; choices: {MODEL_NAMES}")

    estimator = MODEL_FACTORIES[model_name](seed)
    pipeline = build_feature_pipeline(k=k)
    pipeline.steps.append(("clf", estimator))
    return pipeline
