"""Tests for baseline pipeline construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomedical_ml.models import MODEL_NAMES, build_feature_pipeline, build_pipeline


@pytest.fixture
def toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 10))
    y = (X[:, 0] + rng.normal(scale=0.2, size=40) > 0).astype(int)
    return X, y


def test_every_registered_model_builds_a_three_step_pipeline():
    for name in MODEL_NAMES:
        pipeline = build_pipeline(name)
        assert list(pipeline.named_steps) == ["select", "scale", "clf"]


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("not_a_real_model")


@pytest.mark.parametrize(
    ("model_name", "expected_l1_ratio", "expected_solver"),
    [
        ("logreg_l2", 0.0, "lbfgs"),
        ("logreg_l1", 1.0, "saga"),
        ("logreg_elasticnet", 0.5, "saga"),
    ],
)
def test_logistic_regression_variants_use_l1_ratio_not_penalty(
    model_name, expected_l1_ratio, expected_solver
):
    clf = build_pipeline(model_name).named_steps["clf"]

    # The current sklearn API deprecates `penalty=`; regularisation type is
    # controlled by `l1_ratio` alone, so that is what must be asserted here.
    assert clf.l1_ratio == expected_l1_ratio
    assert clf.solver == expected_solver
    assert clf.class_weight == "balanced"


def test_random_forest_and_xgboost_are_class_balanced_or_balanceable():
    rf = build_pipeline("random_forest").named_steps["clf"]
    xgb = build_pipeline("xgboost").named_steps["clf"]

    assert rf.class_weight == "balanced"
    # XGBoost has no class_weight; scale_pos_weight is set per-fold by the
    # evaluation loop instead (see test_evaluation.py).
    assert hasattr(xgb, "scale_pos_weight")


def test_select_k_best_keeps_exactly_k_features_after_fit(toy_data):
    X, y = toy_data
    pipeline = build_pipeline("logreg_l2", k=4)

    pipeline.fit(X, y)

    assert pipeline.named_steps["select"].get_support().sum() == 4


def test_k_larger_than_available_features_is_clamped_by_sklearn(toy_data):
    X, y = toy_data
    pipeline = build_pipeline("logreg_l2", k=1000)

    pipeline.fit(X, y)

    assert pipeline.named_steps["select"].get_support().sum() == X.shape[1]


def test_same_seed_gives_identical_fitted_coefficients(toy_data):
    X, y = toy_data
    first = build_pipeline("logreg_l2", k=5, seed=7)
    second = build_pipeline("logreg_l2", k=5, seed=7)

    first.fit(X, y)
    second.fit(X, y)

    np.testing.assert_array_equal(
        first.named_steps["clf"].coef_, second.named_steps["clf"].coef_
    )


def test_build_feature_pipeline_has_no_classifier_step():
    pipeline = build_feature_pipeline(k=4)

    assert list(pipeline.named_steps) == ["select", "scale"]


def test_classifier_pipeline_shares_feature_pipeline_behaviour(toy_data):
    # The autoencoder (Day 3) builds its input space from build_feature_pipeline
    # directly; this pins down that it selects/scales identically to the first
    # two steps of a classifier pipeline built with the same k.
    X, y = toy_data
    feature_only = build_feature_pipeline(k=4)
    classifier = build_pipeline("logreg_l2", k=4, seed=0)

    feature_only.fit(X, y)
    classifier.fit(X, y)

    np.testing.assert_array_equal(
        feature_only.named_steps["select"].get_support(),
        classifier.named_steps["select"].get_support(),
    )


def test_pipeline_accepts_dataframe_input(toy_data):
    X, y = toy_data
    pipeline = build_pipeline("random_forest", k=3, seed=0)
    X_df = pd.DataFrame(X, columns=[f"probe_{i}" for i in range(X.shape[1])])

    pipeline.fit(X_df, y)
    predictions = pipeline.predict_proba(X_df)

    assert predictions.shape == (40, 2)
