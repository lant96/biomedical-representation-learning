"""SHAP interpretability for the classical baselines.

SHAP, not permutation importance: it gives signed, per-sample, per-feature
attributions, which is what later gets compared against the genes loading the
autoencoder's latent space (Day 4). Permutation importance only gives an
unsigned, dataset-level ranking.

A pipeline here is ``SelectKBest -> StandardScaler -> classifier``. SHAP must
explain the *classifier*, which never sees the raw probes — only the
``k`` selected, scaled features — so every function in this module runs the
upstream transform first and reports results in that reduced feature space
(mapped back to probe/gene names via :func:`selected_feature_names`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

#: Classifiers SHAP can explain exactly and cheaply via their tree structure.
_TREE_MODELS = {"RandomForestClassifier", "XGBClassifier"}
_LINEAR_MODELS = {"LogisticRegression"}


def selected_feature_names(pipeline: Pipeline, feature_names: pd.Index) -> pd.Index:
    """Feature names the pipeline's ``SelectKBest`` step kept, in output order."""
    mask = pipeline.named_steps["select"].get_support()
    return feature_names[mask]


def _build_explainer(clf, background: np.ndarray):
    model_type = type(clf).__name__
    if model_type in _TREE_MODELS:
        return shap.TreeExplainer(clf)
    if model_type in _LINEAR_MODELS:
        return shap.LinearExplainer(clf, background)
    # Fall back to the model-agnostic explainer for anything else. It needs a
    # callable that returns per-class probabilities, not the raw estimator —
    # shap.Explainer(clf, background) raises TypeError on a plain classifier.
    return shap.Explainer(clf.predict_proba, background)


def _positive_class_values(raw_values) -> np.ndarray:
    """Normalise explainer output to one ``(n_samples, n_features)`` array for the SLE class.

    ``TreeExplainer`` on a two-output classifier (e.g. ``RandomForestClassifier``)
    returns shape ``(n, k, 2)``; XGBoost and linear models are inherently
    single-output and already return ``(n, k)``.
    """
    if isinstance(raw_values, list):
        return np.asarray(raw_values[-1])
    values = np.asarray(raw_values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def compute_shap_values(
    pipeline: Pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    *,
    max_background: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, pd.Index]:
    """SHAP values for ``pipeline``'s classifier, on already-selected features.

    Args:
        pipeline: A fitted pipeline from :func:`biomedical_ml.models.build_pipeline`.
        X_background: Reference data the explainer estimates feature impact
            against (typically the training fold). Subsampled to
            ``max_background`` rows for speed on the non-tree explainers.
        X_explain: The samples to explain (typically the held-out fold).

    Returns:
        ``(shap_values, feature_names)`` where ``shap_values`` has shape
        ``(len(X_explain), len(feature_names))`` and ``feature_names`` are the
        probe IDs the classifier actually saw.
    """
    selected = selected_feature_names(pipeline, X_background.columns)
    upstream = pipeline[:-1]  # SelectKBest + StandardScaler, already fitted
    clf = pipeline.named_steps["clf"]

    # Transform via the DataFrame (not .to_numpy()) so SelectKBest sees the
    # feature names it was fitted with, rather than warning that they're missing.
    background = upstream.transform(X_background)
    explain = upstream.transform(X_explain)

    if background.shape[0] > max_background:
        rng = np.random.default_rng(seed)
        idx = rng.choice(background.shape[0], size=max_background, replace=False)
        background = background[idx]

    # The unified __call__ API (rather than the older .shap_values() method)
    # works identically across TreeExplainer, LinearExplainer, and the generic
    # fallback -- .shap_values() isn't implemented on the fallback at all.
    explainer = _build_explainer(clf, background)
    raw_values = explainer(explain).values

    return _positive_class_values(raw_values), selected


def top_shap_genes(
    shap_values: np.ndarray,
    feature_names: pd.Index,
    annotation: pd.DataFrame | None = None,
    *,
    n: int = 25,
) -> pd.DataFrame:
    """Rank features by mean |SHAP|, optionally attaching gene symbols.

    Mean absolute value (rather than mean signed value) is the standard SHAP
    importance ranking: a feature that pushes strongly toward SLE for some
    patients and toward control for others is influential even though its
    signed average could land near zero.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)

    ranking = pd.DataFrame(
        {
            "probe_id": feature_names,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    if annotation is not None:
        ranking["gene_symbol"] = ranking["probe_id"].map(annotation["gene_symbol"])

    return ranking.head(n).reset_index(drop=True)
