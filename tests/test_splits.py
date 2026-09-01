"""Tests for subject-aware splitting.

The point of these is narrow but important: no subject may ever appear on both
sides of a split, because GSE138458 samples many patients more than once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomedical_ml.splits import (
    assert_no_subject_leakage,
    cv_splitter,
    describe_split,
    holdout_split,
    repeated_cv_splits,
)


@pytest.fixture
def cohort() -> tuple[pd.Series, pd.Series]:
    """A cohort with the same awkward shape as the real one: repeats, few controls."""
    rng = np.random.default_rng(0)
    subjects, labels = [], []
    for subject in range(60):  # cases, most sampled twice
        visits = 2 if subject % 2 == 0 else 1
        subjects += [f"case{subject}"] * visits
        labels += [1] * visits
    for subject in range(12):  # controls, mostly single visits
        visits = 2 if subject % 6 == 0 else 1
        subjects += [f"ctrl{subject}"] * visits
        labels += [0] * visits

    order = rng.permutation(len(subjects))
    groups = pd.Series([subjects[i] for i in order], name="subject_id")
    y = pd.Series([labels[i] for i in order], name="sle")
    return y, groups


def test_holdout_split_has_no_subject_leakage(cohort):
    y, groups = cohort
    train_idx, test_idx = holdout_split(y, groups)

    assert_no_subject_leakage(groups, train_idx, test_idx)


def test_holdout_split_covers_every_sample_exactly_once(cohort):
    y, groups = cohort
    train_idx, test_idx = holdout_split(y, groups)

    assert len(train_idx) + len(test_idx) == len(y)
    assert not set(train_idx) & set(test_idx)


def test_holdout_split_keeps_controls_on_both_sides(cohort):
    y, groups = cohort
    train_idx, test_idx = holdout_split(y, groups)

    assert (y.iloc[train_idx] == 0).sum() > 0
    assert (y.iloc[test_idx] == 0).sum() > 0


def test_every_cv_fold_is_subject_disjoint(cohort):
    y, groups = cohort

    for train_idx, test_idx in cv_splitter(n_splits=5).split(np.zeros(len(y)), y, groups):
        assert_no_subject_leakage(groups, train_idx, test_idx)


def test_repeated_cv_yields_all_folds_and_stays_disjoint(cohort):
    y, groups = cohort

    splits = list(repeated_cv_splits(y, groups, n_splits=5, n_repeats=3))

    assert len(splits) == 15
    for train_idx, test_idx in splits:
        assert_no_subject_leakage(groups, train_idx, test_idx)


def test_repeated_cv_actually_varies_between_repeats(cohort):
    y, groups = cohort

    splits = list(repeated_cv_splits(y, groups, n_splits=5, n_repeats=2))
    first_repeat = {tuple(sorted(test)) for _, test in splits[:5]}
    second_repeat = {tuple(sorted(test)) for _, test in splits[5:]}

    # Re-seeding must reshuffle; identical repeats would give false precision.
    assert first_repeat != second_repeat


def test_assert_no_subject_leakage_detects_a_planted_overlap(cohort):
    _, groups = cohort
    shared = groups.index[groups == groups.iloc[0]].to_numpy()

    with pytest.raises(AssertionError, match="both train and test"):
        assert_no_subject_leakage(groups, np.array([shared[0]]), np.array([shared[0]]))


def test_splits_are_reproducible_across_calls(cohort):
    y, groups = cohort

    first = holdout_split(y, groups, seed=7)
    second = holdout_split(y, groups, seed=7)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_describe_split_reports_subject_counts(cohort):
    y, groups = cohort
    train_idx, test_idx = holdout_split(y, groups)

    described = describe_split(y, groups, train_idx, test_idx)

    assert list(described.index) == ["train", "test"]
    assert described.loc["train", "samples"] == len(train_idx)
    assert described["subjects"].sum() == groups.nunique()