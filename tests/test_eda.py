"""Tests for the EDA summary helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomedical_ml.eda import (
    batch_confound_report,
    cohort_summary,
    correlate_features_with_target,
    correlate_pcs_with_batch,
    pca_embedding,
    probe_variance,
)
from biomedical_ml.preprocessing import Dataset


@pytest.fixture
def dataset() -> Dataset:
    rng = np.random.default_rng(0)
    samples = [f"GSM{i}" for i in range(8)]
    X = pd.DataFrame(
        rng.normal(8.0, 1.0, size=(8, 5)),
        index=pd.Index(samples, name="geo_accession"),
        columns=[f"ILMN_{i}" for i in range(5)],
    )
    # Subject "0" is sampled twice; the rest once.
    groups = pd.Series(["0", "0", "1", "2", "3", "4", "5", "6"], index=samples, name="subject_id")
    y = pd.Series([1, 1, 1, 1, 1, 1, 0, 0], index=samples, name="sle")
    metadata = pd.DataFrame(
        {"chip_id": ["chipA"] * 4 + ["chipB"] * 4},
        index=pd.Index(samples, name="geo_accession"),
    )
    return Dataset(X=X, y=y, groups=groups, metadata=metadata)


def test_cohort_summary_counts_samples_and_subjects(dataset):
    summary = cohort_summary(dataset)

    assert summary["n_samples"] == 8
    assert summary["n_subjects"] == 7
    assert summary["n_sle_samples"] == 6
    assert summary["n_control_samples"] == 2
    assert summary["n_control_subjects"] == 2


def test_cohort_summary_reports_visit_distribution(dataset):
    summary = cohort_summary(dataset)

    # Six subjects with one sample, one subject with two.
    assert summary["samples_per_subject"] == {1: 6, 2: 1}


def test_probe_variance_is_sorted_descending(dataset):
    variance = probe_variance(dataset.X)

    assert len(variance) == dataset.X.shape[1]
    assert variance.is_monotonic_decreasing


def test_pca_embedding_shape_and_variance_ratio(dataset):
    scores, ratio = pca_embedding(dataset.X, n_components=3)

    assert scores.shape == (8, 3)
    assert list(scores.columns) == ["PC1", "PC2", "PC3"]
    assert list(scores.index) == list(dataset.X.index)
    assert ratio.sum() <= 1.0 + 1e-9
    # Components are ordered by the variance they explain.
    assert np.all(np.diff(ratio) <= 1e-9)


def test_pca_embedding_caps_components_at_matrix_rank(dataset):
    scores, _ = pca_embedding(dataset.X, n_components=50)

    assert scores.shape[1] <= min(dataset.X.shape)


def test_batch_confound_report_flags_mixed_chips(dataset):
    report = batch_confound_report(dataset.metadata, dataset.y)

    assert report["n_chips"] == 2
    # Only chipB carries controls, and it also carries cases.
    assert report["n_chips_with_control"] == 1
    assert report["max_controls_on_one_chip"] == 2
    assert report["chips_are_mixed"] is True


def test_batch_confound_report_detects_a_pure_control_chip(dataset):
    # A chip holding only controls is the confound this report exists to catch.
    metadata = dataset.metadata.copy()
    metadata["chip_id"] = ["chipA"] * 6 + ["chipB"] * 2

    report = batch_confound_report(metadata, dataset.y)

    assert report["chips_are_mixed"] is False


def test_correlate_features_ranks_the_most_correlated_column_first():
    rng = np.random.default_rng(0)
    target = rng.normal(size=100)
    X = pd.DataFrame(
        {
            "strong": target * 2 + rng.normal(scale=0.1, size=100),
            "weak": target * 0.1 + rng.normal(scale=1.0, size=100),
            "noise": rng.normal(size=100),
        }
    )

    ranking = correlate_features_with_target(X, target)

    assert ranking.iloc[0]["probe_id"] == "strong"
    assert ranking["abs_correlation"].is_monotonic_decreasing


def test_correlate_features_ranks_by_absolute_value_for_negative_correlation():
    rng = np.random.default_rng(0)
    target = rng.normal(size=100)
    X = pd.DataFrame({"anti_correlated": -target * 3 + rng.normal(scale=0.1, size=100)})

    ranking = correlate_features_with_target(X, target)

    assert ranking.iloc[0]["correlation"] < 0
    assert ranking.iloc[0]["abs_correlation"] > 0.9


def test_correlate_features_n_truncates_the_ranking():
    rng = np.random.default_rng(0)
    target = rng.normal(size=50)
    X = pd.DataFrame(rng.normal(size=(50, 10)), columns=[f"probe_{i}" for i in range(10)])

    ranking = correlate_features_with_target(X, target, n=3)

    assert len(ranking) == 3


def test_correlate_features_attaches_gene_symbol_when_annotation_given():
    rng = np.random.default_rng(0)
    target = rng.normal(size=30)
    X = pd.DataFrame(
        {"probeA": target + rng.normal(scale=0.1, size=30), "probeB": rng.normal(size=30)}
    )
    annotation = pd.DataFrame(
        {"gene_symbol": ["GENE_A", "GENE_B"]}, index=pd.Index(["probeA", "probeB"])
    )

    ranking = correlate_features_with_target(X, target, annotation)

    assert ranking.set_index("probe_id").loc["probeA", "gene_symbol"] == "GENE_A"


def test_correlate_features_without_annotation_omits_gene_symbol_column():
    rng = np.random.default_rng(0)
    target = rng.normal(size=20)
    X = pd.DataFrame({"probeA": rng.normal(size=20)})

    ranking = correlate_features_with_target(X, target, annotation=None)

    assert "gene_symbol" not in ranking.columns


def test_correlate_features_accepts_a_pandas_series_target():
    rng = np.random.default_rng(0)
    target = pd.Series(rng.normal(size=40), name="latent_dim")
    X = pd.DataFrame({"probeA": target.to_numpy() + rng.normal(scale=0.1, size=40)})

    ranking = correlate_features_with_target(X, target)

    assert ranking.iloc[0]["abs_correlation"] > 0.9


@pytest.fixture
def chip_scores_and_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """24 samples on 3 chips (8 each); PC1 tracks chip, PC2 does not."""
    rng = np.random.default_rng(0)
    n_per_chip = 8
    chips = ["chipA"] * n_per_chip + ["chipB"] * n_per_chip + ["chipC"] * n_per_chip
    chip_means = {"chipA": 10.0, "chipB": 0.0, "chipC": -10.0}

    samples = [f"GSM{i}" for i in range(len(chips))]
    pc1 = np.array([chip_means[c] for c in chips]) + rng.normal(scale=0.5, size=len(chips))
    pc2 = rng.normal(size=len(chips))  # unrelated to chip

    scores = pd.DataFrame({"PC1": pc1, "PC2": pc2}, index=pd.Index(samples, name="geo_accession"))
    metadata = pd.DataFrame({"chip_id": chips}, index=scores.index)
    return scores, metadata


def test_correlate_pcs_with_batch_flags_the_batch_associated_component(chip_scores_and_metadata):
    scores, metadata = chip_scores_and_metadata

    result = correlate_pcs_with_batch(scores, metadata, n_components=2)

    assert result.loc["PC1", "eta_squared"] > 0.9
    assert result.loc["PC1", "p_value"] < 0.001
    assert result.loc["PC2", "eta_squared"] < result.loc["PC1", "eta_squared"]


def test_correlate_pcs_with_batch_returns_expected_columns(chip_scores_and_metadata):
    scores, metadata = chip_scores_and_metadata

    result = correlate_pcs_with_batch(scores, metadata, n_components=2)

    assert list(result.columns) == ["f_stat", "p_value", "eta_squared"]
    assert result.index.name == "component"


def test_correlate_pcs_with_batch_handles_all_singleton_chips():
    # Every chip has exactly one sample, so every PC gets skipped (fewer than
    # two multi-sample groups) -- must return an empty frame, not crash.
    samples = [f"GSM{i}" for i in range(4)]
    scores = pd.DataFrame(
        {"PC1": [1.0, 2.0, 3.0, 4.0]}, index=pd.Index(samples, name="geo_accession")
    )
    metadata = pd.DataFrame({"chip_id": [f"chip{i}" for i in range(4)]}, index=scores.index)

    result = correlate_pcs_with_batch(scores, metadata, n_components=1)

    assert result.empty
    assert list(result.columns) == ["f_stat", "p_value", "eta_squared"]
    assert result.index.name == "component"