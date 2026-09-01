"""Tests for the GEO series-matrix parser.

These run against a small synthetic matrix rather than the real download, so the
suite stays fast and works offline.
"""

from __future__ import annotations

import gzip
import ssl
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from biomedical_ml.data import (
    _download,
    _ssl_context,
    parse_platform_annotation,
    parse_series_matrix,
)

MATRIX = """\
!Series_title\t"A small study"
!Sample_title\t"ctrl 1"\t"case 1"\t"case 2"
!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"
!Sample_source_name_ch1\t"whole blood"\t"whole blood"\t"whole blood"
!Sample_characteristics_ch1\t"disease state: healthy"\t"disease state: SLE"\t"disease state: SLE"
!Sample_characteristics_ch1\t"Sex: F"\t"Sex: M"\t"Sex: F"
!series_matrix_table_begin
"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"
"ILMN_1"\t5.1\t6.2\t7.3
"ILMN_2"\t1.0\t2.0\t3.0
!series_matrix_table_end
"""


@pytest.fixture
def matrix_path(tmp_path: Path) -> Path:
    path = tmp_path / "GSE_test_series_matrix.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(MATRIX)
    return path


def test_expression_is_probes_by_samples(matrix_path: Path) -> None:
    expression, _ = parse_series_matrix(matrix_path)

    assert expression.shape == (2, 3)
    assert expression.index.tolist() == ["ILMN_1", "ILMN_2"]
    assert expression.columns.tolist() == ["GSM1", "GSM2", "GSM3"]


def test_expression_values_are_numeric(matrix_path: Path) -> None:
    expression, _ = parse_series_matrix(matrix_path)

    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in expression.dtypes)
    assert expression.loc["ILMN_1", "GSM2"] == pytest.approx(6.2)


def test_characteristics_become_columns(matrix_path: Path) -> None:
    _, metadata = parse_series_matrix(matrix_path)

    assert metadata.index.tolist() == ["GSM1", "GSM2", "GSM3"]
    assert metadata.loc["GSM1", "disease_state"] == "healthy"
    assert metadata.loc["GSM2", "disease_state"] == "SLE"
    assert metadata.loc["GSM3", "sex"] == "F"


def test_scalar_sample_fields_are_kept(matrix_path: Path) -> None:
    _, metadata = parse_series_matrix(matrix_path)

    assert metadata.loc["GSM2", "title"] == "case 1"
    assert metadata.loc["GSM2", "source_name_ch1"] == "whole blood"


def test_characteristic_keys_may_be_ordered_differently_per_sample(tmp_path: Path) -> None:
    # GEO submitters sometimes emit characteristics in an inconsistent order;
    # values must follow their key, not the row they arrived on.
    shuffled = MATRIX.replace(
        '!Sample_characteristics_ch1\t"disease state: healthy"\t"disease state: SLE"'
        '\t"disease state: SLE"\n'
        '!Sample_characteristics_ch1\t"Sex: F"\t"Sex: M"\t"Sex: F"',
        '!Sample_characteristics_ch1\t"disease state: healthy"\t"Sex: M"\t"disease state: SLE"\n'
        '!Sample_characteristics_ch1\t"Sex: F"\t"disease state: SLE"\t"Sex: F"',
    )
    path = tmp_path / "shuffled_series_matrix.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(shuffled)

    _, metadata = parse_series_matrix(path)

    assert metadata.loc["GSM2", "disease_state"] == "SLE"
    assert metadata.loc["GSM2", "sex"] == "M"


def test_missing_accession_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad_series_matrix.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(MATRIX.replace('!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"\n', ""))

    with pytest.raises(ValueError, match="geo_accession"):
        parse_series_matrix(path)


def test_missing_table_raises(tmp_path: Path) -> None:
    path = tmp_path / "no_table_series_matrix.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(MATRIX.replace("!series_matrix_table_begin\n", ""))

    with pytest.raises(ValueError, match="expression table"):
        parse_series_matrix(path)


def test_characteristic_cell_without_a_colon_is_ignored(tmp_path: Path) -> None:
    # A characteristics_ch1 cell with no "key: value" shape has no key to file
    # it under; it should be skipped rather than crash or corrupt other keys.
    malformed = MATRIX.replace(
        '!Sample_characteristics_ch1\t"Sex: F"\t"Sex: M"\t"Sex: F"',
        '!Sample_characteristics_ch1\t"Sex: F"\t"just some text"\t"Sex: F"',
    )
    path = tmp_path / "malformed_series_matrix.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(malformed)

    _, metadata = parse_series_matrix(path)

    assert metadata.loc["GSM1", "sex"] == "F"
    assert metadata.loc["GSM3", "sex"] == "F"
    assert pd.isna(metadata.loc["GSM2", "sex"])


ANNOTATION = """\
#ID = probe identifier
!platform_table_begin
ID\tGene title\tGene symbol\tGene ID
ILMN_1\teukaryotic translation elongation factor 1 alpha 1\tEEF1A1\t1915
ILMN_2\t\t\t
!platform_table_end
"""


@pytest.fixture
def annotation_path(tmp_path: Path) -> Path:
    path = tmp_path / "GPL_test.annot.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(ANNOTATION)
    return path


def test_parse_platform_annotation_maps_probes_to_genes(annotation_path: Path) -> None:
    annotation = parse_platform_annotation(annotation_path)

    assert annotation.index.name == "probe_id"
    assert annotation.loc["ILMN_1", "gene_symbol"] == "EEF1A1"
    assert list(annotation.columns) == ["gene_title", "gene_symbol", "gene_id"]


def test_parse_platform_annotation_leaves_unmapped_probes_as_nan(annotation_path: Path) -> None:
    annotation = parse_platform_annotation(annotation_path)

    assert pd.isna(annotation.loc["ILMN_2", "gene_symbol"])


def test_parse_platform_annotation_missing_table_raises(tmp_path: Path) -> None:
    path = tmp_path / "no_table.annot.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(ANNOTATION.replace("!platform_table_begin\n", ""))

    with pytest.raises(ValueError, match="platform table"):
        parse_platform_annotation(path)


def test_ssl_context_returns_a_real_ssl_context() -> None:
    context = _ssl_context()

    assert isinstance(context, ssl.SSLContext)


def test_ssl_context_falls_back_when_truststore_is_unavailable() -> None:
    # Simulate truststore not being installed: sys.modules[name] = None makes
    # `import truststore` raise ImportError, exercising the fallback branch.
    with patch.dict(sys.modules, {"truststore": None}):
        context = _ssl_context()

    assert isinstance(context, ssl.SSLContext)


def test_download_skips_when_already_cached(tmp_path: Path) -> None:
    dest = tmp_path / "cached.txt"
    dest.write_text("already here", encoding="utf-8")

    with patch("urllib.request.urlopen") as mock_urlopen:
        result = _download("https://example.invalid/file", dest)

    mock_urlopen.assert_not_called()
    assert result == dest
    assert dest.read_text(encoding="utf-8") == "already here"


def test_download_force_redownloads_even_if_cached(tmp_path: Path) -> None:
    dest = tmp_path / "cached.txt"
    dest.write_text("stale", encoding="utf-8")

    # io.BytesIO already implements the context-manager and .read(size)
    # protocol shutil.copyfileobj needs, EOF handling included.
    with patch("urllib.request.urlopen", return_value=BytesIO(b"fresh")):
        _download("https://example.invalid/file", dest, force=True)

    assert dest.read_bytes() == b"fresh"


def test_download_writes_to_final_path_not_left_as_partial(tmp_path: Path) -> None:
    dest = tmp_path / "new_download.txt"

    with patch("urllib.request.urlopen", return_value=BytesIO(b"content")):
        result = _download("https://example.invalid/file", dest)

    assert result == dest
    assert dest.exists()
    assert dest.read_bytes() == b"content"
    assert not dest.with_suffix(dest.suffix + ".part").exists()
