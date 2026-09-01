"""Tests for repeated grouped CV evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from biomedical_ml.evaluation import (
    METRICS,
    _best_threshold,
    _set_xgboost_class_balance,
    best_model,
    evaluate_model,
    evaluate_models,
    evaluate_pipeline_factory,
    summarize_results,
)
from biomedical_ml.models import build_pipeline


@pytest.fixture
def cohort_with_signal() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """A small cohort shaped like GSE138458: repeat visits, imbalanced, with signal.

    Mirrors the repeat-visit / imbalance structure used in test_splits.py's
    ``cohort`` fixture, but adds an actual informative feature so ROC-AUC has
    something other than 0.5 to report.
    """
    rng = np.random.default_rng(0)
    subjects, labels = [], []
    for subject in range(24):  # cases, most sampled twice
        visits = 2 if subject % 2 == 0 else 1
        subjects += [f"case{subject}"] * visits
        labels += [1] * visits
    for subject in range(8):  # controls, mostly single visits
        visits = 2 if subject % 4 == 0 else 1
        subjects += [f"ctrl{subject}"] * visits
        labels += [0] * visits

    order = rng.permutation(len(subjects))
    groups = pd.Series([subjects[i] for i in order], name="subject_id")
    y = pd.Series([labels[i] for i in order], name="sle")

    n = len(y)
    informative = y.to_numpy() * 2.0 + rng.normal(scale=1.0, size=n)
    noise = rng.normal(size=(n, 9))
    X = pd.DataFrame(
        np.column_stack([informative, noise]),
        columns=["informative", *[f"noise{i}" for i in range(9)]],
    )
    return X, y, groups


def test_evaluate_model_returns_one_row_per_fold(cohort_with_signal):
    X, y, groups = cohort_with_signal

    results = evaluate_model("logreg_l2", X, y, groups, k=5, n_splits=3, n_repeats=2, seed=0)

    assert len(results) == 6
    assert set(results["model"]) == {"logreg_l2"}
    assert set(zip(results["repeat"], results["fold"], strict=True)) == {
        (r, f) for r in range(2) for f in range(3)
    }


def test_evaluate_model_metrics_are_within_valid_ranges(cohort_with_signal):
    X, y, groups = cohort_with_signal

    results = evaluate_model("random_forest", X, y, groups, k=5, n_splits=3, n_repeats=2, seed=0)

    for metric in METRICS:
        assert results[metric].between(0, 1).all(), f"{metric} out of [0, 1] range"


def test_evaluate_model_beats_chance_when_signal_is_present(cohort_with_signal):
    # A sanity check on the whole pipeline: with an informative feature this
    # strong, a linear model should clear a random-guessing baseline.
    X, y, groups = cohort_with_signal

    results = evaluate_model("logreg_l2", X, y, groups, k=5, n_splits=3, n_repeats=2, seed=0)

    assert results["roc_auc"].mean() > 0.7


def test_evaluate_models_concatenates_one_block_per_model(cohort_with_signal):
    X, y, groups = cohort_with_signal

    results = evaluate_models(
        ["logreg_l2", "random_forest"], X, y, groups, k=5, n_splits=3, n_repeats=2, seed=0
    )

    assert set(results["model"]) == {"logreg_l2", "random_forest"}
    assert len(results) == 2 * 3 * 2


def test_best_threshold_beats_naive_half_under_heavy_imbalance():
    # 18 of the 20 "positive" labels sit just above 0.5; a blind 0.5 cutoff
    # would correctly catch them, but the 2 negatives sit at low probabilities
    # that only a lower threshold captures. Balanced accuracy should reward
    # finding a threshold that catches both negatives without losing positives.
    y = np.array([1] * 18 + [0] * 2)
    proba = np.array([0.55] * 18 + [0.05, 0.05])

    threshold = _best_threshold(y, proba)
    pred = proba >= threshold

    assert balanced_accuracy_score(y, pred) == pytest.approx(1.0)


def test_best_threshold_is_bounded_between_zero_and_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=30)
    proba = rng.uniform(size=30)

    threshold = _best_threshold(y, proba)

    assert 0.0 <= threshold <= 1.0


def test_evaluate_pipeline_factory_accepts_a_non_classical_pipeline(cohort_with_signal):
    # Day 4's use case: a plain scale+logreg pipeline on already-dense features
    # (no SelectKBest), which evaluate_model can't build since it always goes
    # through models.build_pipeline.
    X, y, groups = cohort_with_signal

    def probe_factory(seed: int) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
            ]
        )

    results = evaluate_pipeline_factory(
        "toy_probe", probe_factory, X, y, groups, n_splits=3, n_repeats=2, seed=0
    )

    assert set(results["model"]) == {"toy_probe"}
    assert len(results) == 6
    for metric in METRICS:
        assert results[metric].between(0, 1).all()


def test_evaluate_model_is_a_thin_wrapper_over_pipeline_factory(cohort_with_signal):
    # evaluate_model must still produce identical output after the refactor
    # that routes it through evaluate_pipeline_factory.
    X, y, groups = cohort_with_signal

    direct = evaluate_model("logreg_l2", X, y, groups, k=5, n_splits=3, n_repeats=2, seed=0)
    via_factory = evaluate_pipeline_factory(
        "logreg_l2",
        lambda s: build_pipeline("logreg_l2", k=5, seed=s),
        X,
        y,
        groups,
        n_splits=3,
        n_repeats=2,
        seed=0,
    )

    pd.testing.assert_frame_equal(direct, via_factory)


def test_set_xgboost_class_balance_uses_neg_over_pos_ratio():
    pipeline = build_pipeline("xgboost", k=3, seed=0)
    y_train = np.array([1, 1, 1, 1, 0])  # 4 positive, 1 negative

    _set_xgboost_class_balance(pipeline, y_train)

    assert pipeline.named_steps["clf"].scale_pos_weight == pytest.approx(0.25)


def test_set_xgboost_class_balance_is_a_noop_for_other_models():
    pipeline = build_pipeline("logreg_l2", k=3, seed=0)
    y_train = np.array([1, 1, 0])

    _set_xgboost_class_balance(pipeline, y_train)  # must not raise

    assert not hasattr(pipeline.named_steps["clf"], "scale_pos_weight")


def test_summarize_results_ranks_by_mean_roc_auc():
    results = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "roc_auc": [0.6, 0.7, 0.9, 0.95],
            "pr_auc": [0.5, 0.5, 0.5, 0.5],
            "f1": [0.5, 0.5, 0.5, 0.5],
            "balanced_accuracy": [0.5, 0.5, 0.5, 0.5],
        }
    )

    summary = summarize_results(results)

    assert list(summary.index) == ["b", "a"]
    assert summary.loc["a", "roc_auc_mean"] == pytest.approx(0.65)
    assert summary.loc["b", "roc_auc_mean"] == pytest.approx(0.925)


def test_best_model_returns_top_of_summary():
    results = pd.DataFrame(
        {
            "model": ["a", "b"],
            "roc_auc": [0.6, 0.9],
            "pr_auc": [0.5, 0.5],
            "f1": [0.5, 0.5],
            "balanced_accuracy": [0.5, 0.5],
        }
    )

    assert best_model(results) == "b"
