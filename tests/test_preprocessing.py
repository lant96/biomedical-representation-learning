"""Tests for dataset assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import requires_real_data

from biomedical_ml import preprocessing
from biomedical_ml.preprocessing import (
    LABEL_MAP,
    add_batch_columns,
    build_dataset,
    drop_empty_samples,
)


def test_drop_empty_samples_removes_all_nan_columns(synthetic_expression, synthetic_metadata):
    expression, metadata = drop_empty_samples(synthetic_expression, synthetic_metadata)

    assert expression.shape[1] == 8
    assert "GSM8" not in expression.columns
    assert "GSM9" not in expression.columns
    # Metadata must be dropped in lockstep, or labels misalign with samples.
    assert list(metadata.index) == list(expression.columns)


def test_drop_empty_samples_keeps_partially_missing_columns(
    synthetic_expression, synthetic_metadata
):
    # Only *entirely* empty samples are placeholders; a single missing probe
    # is ordinary and must not cost us the sample.
    synthetic_expression.loc["ILMN_0", "GSM3"] = pd.NA

    expression, _ = drop_empty_samples(synthetic_expression, synthetic_metadata)

    assert "GSM3" in expression.columns


def test_add_batch_columns_splits_chip_and_position(synthetic_metadata):
    result = add_batch_columns(synthetic_metadata)

    assert result.loc["GSM0", "chip_id"] == "200319680110"
    assert result.loc["GSM0", "array_position"] == "A"
    assert result.loc["GSM5", "array_position"] == "B"


def test_label_map_is_binary_with_sle_positive():
    assert LABEL_MAP["SLE Case"] == 1
    assert LABEL_MAP["Control"] == 0


def test_dataset_summary_reports_samples_subjects_and_class_split():
    samples = ["GSM0", "GSM1", "GSM2", "GSM3"]
    X = pd.DataFrame({"g1": [1, 2, 3, 4]}, index=pd.Index(samples, name="geo_accession"))
    y = pd.Series([1, 1, 0, 0], index=X.index, name="sle")
    groups = pd.Series(["s0", "s0", "s1", "s2"], index=X.index, name="subject_id")
    dataset = preprocessing.Dataset(X=X, y=y, groups=groups, metadata=pd.DataFrame(index=X.index))

    assert dataset.n_subjects == 3
    text = dataset.summary()
    assert "4 samples x 1 probes" in text
    assert "from 3 subjects" in text
    assert "SLE:        2 samples from   1 subjects" in text
    assert "Control:    2 samples from   2 subjects" in text


def test_build_dataset_raises_on_missing_subject_id(
    monkeypatch, synthetic_expression, synthetic_metadata
):
    metadata = synthetic_metadata.copy()
    metadata.loc[metadata.index[0], "subject_id"] = np.nan

    def fake_load(raw_dir):
        return synthetic_expression, metadata

    monkeypatch.setattr(preprocessing, "load_series_matrix", fake_load)

    with pytest.raises(ValueError, match="Missing subject_id"):
        build_dataset(annotated_only=False, with_annotation=False)


def test_build_dataset_raises_on_literal_nan_string_subject_id(
    monkeypatch, synthetic_expression, synthetic_metadata
):
    # Distinct from an actual missing value: the source data already contains
    # the literal text "nan" rather than a null -- astype(str) would otherwise
    # let this slip through as a normal-looking group label.
    metadata = synthetic_metadata.copy()
    metadata.loc[metadata.index[0], "subject_id"] = "nan"

    def fake_load(raw_dir):
        return synthetic_expression, metadata

    monkeypatch.setattr(preprocessing, "load_series_matrix", fake_load)

    with pytest.raises(ValueError, match="Missing subject_id"):
        build_dataset(annotated_only=False, with_annotation=False)


@requires_real_data
def test_real_dataset_has_expected_shape():
    dataset = build_dataset(annotated_only=False, with_annotation=False)

    # 336 GEO samples minus the 6 empty placeholders.
    assert dataset.X.shape == (330, 47323)
    assert int((dataset.y == 1).sum()) == 307
    assert int((dataset.y == 0).sum()) == 23


@requires_real_data
def test_real_dataset_has_repeated_subjects():
    dataset = build_dataset(annotated_only=False, with_annotation=False)

    # Fewer subjects than samples is the whole reason splits must be grouped.
    assert dataset.n_subjects == 218
    assert dataset.n_subjects < dataset.X.shape[0]


@requires_real_data
def test_real_dataset_has_no_missing_values():
    dataset = build_dataset(annotated_only=False, with_annotation=False)

    assert not dataset.X.isna().to_numpy().any()


@requires_real_data
def test_no_subject_is_both_case_and_control():
    dataset = build_dataset(annotated_only=False, with_annotation=False)

    labels_per_subject = dataset.y.groupby(dataset.groups.to_numpy()).nunique()
    assert labels_per_subject.max() == 1


@requires_real_data
def test_annotated_only_keeps_named_genes_and_shrinks_matrix():
    full = build_dataset(annotated_only=False, with_annotation=False)
    named = build_dataset(annotated_only=True)

    assert named.X.shape[1] < full.X.shape[1]
    assert named.annotation is not None
    assert named.annotation["gene_symbol"].notna().all()
    assert list(named.annotation.index) == list(named.X.columns)


@requires_real_data
def test_expression_is_on_log_scale():
    # GEO ships this series already log2-transformed; a raw-intensity matrix
    # would run to five or six figures and need a transform we deliberately omit.
    dataset = build_dataset(annotated_only=True, with_annotation=False)

    assert dataset.X.to_numpy().max() < 32
    assert dataset.X.to_numpy().min() > 0


@requires_real_data
def test_unexpected_label_raises(monkeypatch):
    from biomedical_ml import preprocessing

    original = preprocessing.load_series_matrix

    def corrupted(raw_dir):
        expression, metadata = original(raw_dir)
        metadata = metadata.copy()
        metadata.iloc[0, metadata.columns.get_loc("case_control")] = "Unknown Group"
        return expression, metadata

    monkeypatch.setattr(preprocessing, "load_series_matrix", corrupted)

    with pytest.raises(ValueError, match="Unexpected case_control"):
        build_dataset(annotated_only=False, with_annotation=False)