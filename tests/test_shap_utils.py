"""Tests for the SHAP interpretability helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from biomedical_ml.models import build_pipeline
from biomedical_ml.shap_utils import (
    _build_explainer,
    _positive_class_values,
    compute_shap_values,
    selected_feature_names,
    top_shap_genes,
)


@pytest.fixture
def separable_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """One feature strongly correlated with the label, five pure-noise features.

    ``f_classif`` (used by ``SelectKBest``) should rank ``informative`` far
    above the noise columns, so ``k=1`` deterministically keeps only it —
    which removes any ambiguity about which feature a SHAP sign check is
    talking about.
    """
    rng = np.random.default_rng(0)
    n = 80
    y = rng.integers(0, 2, size=n)
    informative = y * 3.0 + rng.normal(scale=0.3, size=n)
    noise = rng.normal(size=(n, 5))

    X = pd.DataFrame(
        np.column_stack([informative, noise]),
        columns=["informative", *[f"noise{i}" for i in range(5)]],
    )
    return X, pd.Series(y, name="label")


def test_selected_feature_names_matches_select_k_best_support(separable_dataset):
    X, y = separable_dataset
    pipeline = build_pipeline("logreg_l2", k=3, seed=0)
    pipeline.fit(X, y)

    names = selected_feature_names(pipeline, X.columns)
    mask = pipeline.named_steps["select"].get_support()

    assert list(names) == list(X.columns[mask])
    assert len(names) == 3


@pytest.mark.parametrize("model_name", ["logreg_l2", "random_forest", "xgboost"])
def test_compute_shap_values_shape_matches_selected_features(separable_dataset, model_name):
    X, y = separable_dataset
    pipeline = build_pipeline(model_name, k=1, seed=0)
    pipeline.fit(X, y)

    shap_values, feature_names = compute_shap_values(pipeline, X, X)

    assert list(feature_names) == ["informative"]
    assert shap_values.shape == (len(X), 1)


def test_build_explainer_falls_back_to_the_generic_explainer_for_unregistered_models():
    # RandomForestClassifier/XGBClassifier/LogisticRegression all have dedicated
    # branches; anything else must fall through to shap.Explainer. This branch
    # previously passed the raw estimator (shap.Explainer(clf, background)),
    # which raises TypeError -- shap needs a callable like predict_proba.
    rng = np.random.default_rng(0)
    background = rng.normal(size=(10, 3))
    clf = KNeighborsClassifier(n_neighbors=3).fit(background, [0, 1] * 5)

    explainer = _build_explainer(clf, background)

    assert type(explainer).__module__.startswith("shap.")


def test_compute_shap_values_works_for_a_model_outside_the_registry(separable_dataset):
    # Exercises the fallback explainer end-to-end, on a plain sklearn pipeline
    # rather than one of models.MODEL_NAMES (KNeighborsClassifier is
    # deliberately not a registered baseline).
    X, y = separable_dataset
    pipeline = Pipeline(
        [
            ("select", SelectKBest(f_classif, k=1)),
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=3)),
        ]
    )
    pipeline.fit(X, y)

    shap_values, feature_names = compute_shap_values(pipeline, X, X)

    assert list(feature_names) == ["informative"]
    assert shap_values.shape == (len(X), 1)


def test_compute_shap_values_subsamples_large_background_sets(separable_dataset):
    # separable_dataset has 80 rows; max_background=20 forces the subsampling
    # branch. Uses a linear model, since LinearExplainer actually consumes the
    # background array (TreeExplainer ignores it entirely).
    X, y = separable_dataset
    pipeline = build_pipeline("logreg_l2", k=1, seed=0)
    pipeline.fit(X, y)

    shap_values, feature_names = compute_shap_values(pipeline, X, X, max_background=20)

    assert list(feature_names) == ["informative"]
    assert shap_values.shape == (len(X), 1)


@pytest.mark.parametrize("model_name", ["logreg_l2", "random_forest", "xgboost"])
def test_shap_attribution_is_higher_for_class_favouring_feature_values(
    separable_dataset, model_name
):
    X, y = separable_dataset
    pipeline = build_pipeline(model_name, k=1, seed=0)
    pipeline.fit(X, y)

    shap_values, _ = compute_shap_values(pipeline, X, X)

    # Samples where the informative feature sits on the SLE-like (high) side
    # should get a higher attribution, on average, than the control-like side —
    # this holds regardless of each explainer's baseline convention.
    high_mask = (X["informative"] > X["informative"].median()).to_numpy()
    assert shap_values[high_mask].mean() > shap_values[~high_mask].mean()


def test_top_shap_genes_ranks_by_mean_absolute_value(separable_dataset):
    X, y = separable_dataset
    pipeline = build_pipeline("random_forest", k=3, seed=0)
    pipeline.fit(X, y)
    shap_values, feature_names = compute_shap_values(pipeline, X, X)

    ranking = top_shap_genes(shap_values, feature_names, annotation=None, n=2)

    assert len(ranking) == 2
    assert ranking["mean_abs_shap"].is_monotonic_decreasing
    assert "informative" in ranking["probe_id"].to_numpy()


def test_top_shap_genes_attaches_gene_symbol_when_annotation_given(separable_dataset):
    X, y = separable_dataset
    pipeline = build_pipeline("random_forest", k=3, seed=0)
    pipeline.fit(X, y)
    shap_values, feature_names = compute_shap_values(pipeline, X, X)

    annotation = pd.DataFrame(
        {"gene_symbol": [f"GENE_{name}" for name in feature_names]}, index=feature_names
    )

    ranking = top_shap_genes(shap_values, feature_names, annotation, n=3)

    assert "gene_symbol" in ranking.columns
    for _, row in ranking.iterrows():
        assert row["gene_symbol"] == f"GENE_{row['probe_id']}"


def test_top_shap_genes_without_annotation_omits_gene_symbol_column(separable_dataset):
    X, y = separable_dataset
    pipeline = build_pipeline("random_forest", k=3, seed=0)
    pipeline.fit(X, y)
    shap_values, feature_names = compute_shap_values(pipeline, X, X)

    ranking = top_shap_genes(shap_values, feature_names, annotation=None, n=3)

    assert "gene_symbol" not in ranking.columns


def test_positive_class_values_reduces_three_dimensional_tree_output():
    raw = np.zeros((4, 2, 2))
    raw[:, :, 1] = 5.0

    result = _positive_class_values(raw)

    assert result.shape == (4, 2)
    assert (result == 5.0).all()


def test_positive_class_values_passes_through_two_dimensional_output():
    raw = np.ones((4, 2))

    result = _positive_class_values(raw)

    assert np.array_equal(result, raw)


def test_positive_class_values_handles_legacy_list_output():
    raw = [np.zeros((4, 2)), np.ones((4, 2))]

    result = _positive_class_values(raw)

    assert (result == 1).all()
