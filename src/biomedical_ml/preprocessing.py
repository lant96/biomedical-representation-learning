"""Assemble the modelling matrix for GSE138458.

Three properties of this series drive the choices here.

1. Six samples (``data_row_count == 0``) carry no expression data at all — their
   columns are entirely missing. Dropping them leaves 330 samples, 307 SLE and
   23 control.
2. The 330 samples come from only 218 *subjects*: many patients were sampled at
   more than one visit. Sample-level splitting would therefore put the same
   patient on both sides of a train/test boundary, so every split in this
   project groups by ``subject_id`` (see :mod:`biomedical_ml.splits`).
3. GEO ships this series already background-corrected, log2-transformed and
   normalised (values span roughly 6.7-16.4), so no further transform is
   applied. Scaling is left to the per-fold pipeline.

Feature selection is deliberately *not* done here. With 47,323 probes and 330
samples it is tempting to pre-filter the matrix once, but any filter that looks
at expression values leaks test-fold information into training. The only filter
offered at this stage is ``annotated_only``, which uses the platform annotation
rather than the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from biomedical_ml.config import RAW_DIR
from biomedical_ml.data import load_probe_annotation, load_series_matrix

LABEL_COLUMN = "case_control"
GROUP_COLUMN = "subject_id"
BATCH_COLUMN = "chip_id"
POSITIVE_LABEL = "SLE Case"
NEGATIVE_LABEL = "Control"

#: Maps the GEO ``case_control`` strings onto the binary target.
LABEL_MAP = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}


@dataclass(frozen=True)
class Dataset:
    """The modelling matrix and everything needed to split it honestly.

    Attributes:
        X: Expression, samples (rows) by probes (columns) — sklearn orientation.
        y: Binary target, 1 = SLE, 0 = healthy control.
        groups: Subject ID per sample; pass to any grouped CV splitter.
        metadata: The full GEO sample table for the retained samples.
        annotation: Probe-to-gene mapping, or ``None`` if it was not loaded.
    """

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    metadata: pd.DataFrame
    annotation: pd.DataFrame | None = None

    @property
    def n_subjects(self) -> int:
        return int(self.groups.nunique())

    def summary(self) -> str:
        counts = self.y.value_counts()
        by_class = self.groups.groupby(self.y).nunique()
        return (
            f"{self.X.shape[0]} samples x {self.X.shape[1]} probes "
            f"from {self.n_subjects} subjects\n"
            f"  SLE:     {counts.get(1, 0):>4} samples "
            f"from {by_class.get(1, 0):>3} subjects\n"
            f"  Control: {counts.get(0, 0):>4} samples "
            f"from {by_class.get(0, 0):>3} subjects"
        )


def drop_empty_samples(
    expression: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove samples whose expression column is entirely missing.

    GSE138458 contains six such placeholder samples. They are flagged both by
    ``data_row_count == 0`` in the metadata and by an all-NaN expression column;
    we test the data itself, since that is what actually breaks downstream code.
    """
    usable = expression.columns[~expression.isna().all()]
    return expression.loc[:, usable], metadata.loc[usable]


def add_batch_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    """Split the GEO ``description`` field into BeadChip barcode and array position.

    Illumina reports each sample as ``<chip barcode>_<position>``, e.g.
    ``200319680117_A``. The barcode is the only batch variable this series
    exposes, and it is worth having: it is what lets us confirm that the 23
    controls are spread across 23 different chips rather than confounded with a
    single batch.
    """
    metadata = metadata.copy()
    parts = metadata["description"].astype(str).str.split("_", n=1, expand=True)
    metadata[BATCH_COLUMN] = parts[0]
    metadata["array_position"] = parts[1] if parts.shape[1] > 1 else pd.NA
    return metadata


def build_dataset(
    *,
    raw_dir: Path = RAW_DIR,
    annotated_only: bool = False,
    with_annotation: bool = True,
) -> Dataset:
    """Load GSE138458 and assemble it into a :class:`Dataset`.

    Args:
        raw_dir: Where the downloaded GEO files live.
        annotated_only: Keep only probes that map to a named gene. This filter
            reads the platform annotation, not the expression values, so it
            leaks nothing; it drops Illumina control and unmapped probes.
        with_annotation: Attach the probe-to-gene table to the result.
    """
    expression, metadata = load_series_matrix(raw_dir)
    expression, metadata = drop_empty_samples(expression, metadata)
    metadata = add_batch_columns(metadata)

    annotation = load_probe_annotation(raw_dir) if (with_annotation or annotated_only) else None

    if annotated_only:
        assert annotation is not None
        named = annotation["gene_symbol"].notna()
        keep = annotation.index[named].intersection(expression.index)
        expression = expression.loc[keep]

    labels = metadata[LABEL_COLUMN]
    unknown = set(labels.dropna().unique()) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unexpected {LABEL_COLUMN} values: {sorted(unknown)}")

    y = labels.map(LABEL_MAP).astype("int8")
    y.name = "sle"

    groups = metadata[GROUP_COLUMN].astype(str)
    groups.name = "subject_id"

    if groups.isna().any() or (groups == "nan").any():
        raise ValueError("Missing subject_id: grouped splitting would be unsafe")

    # Transpose to sklearn's samples-by-features orientation.
    X = expression.T
    X.index.name = "geo_accession"

    if annotation is not None:
        annotation = annotation.reindex(X.columns)

    return Dataset(X=X, y=y, groups=groups, metadata=metadata, annotation=annotation)