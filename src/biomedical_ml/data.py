"""Download and parse the GSE138458 series matrix from NCBI GEO.

GSE138458 is whole-blood gene expression from an adult SLE cohort plus healthy
controls, assayed on a single platform (GPL10558, Illumina HumanHT-12 v4). Using
one platform is deliberate: it removes the cross-study batch correction that a
multi-cohort aggregator would force on us.

The series matrix is a single text file holding both the sample metadata
(``!Sample_*`` header lines) and the probe-by-sample expression table, so it is
all this project needs. We parse it directly rather than pulling the much larger
SOFT family file.
"""

from __future__ import annotations

import gzip
import re
import shutil
import ssl
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

from biomedical_ml.config import RAW_DIR

GEO_ACCESSION = "GSE138458"
PLATFORM = "GPL10558"

SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE138nnn/"
    f"{GEO_ACCESSION}/matrix/{GEO_ACCESSION}_series_matrix.txt.gz"
)
ANNOTATION_URL = (
    f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL10nnn/{PLATFORM}/annot/{PLATFORM}.annot.gz"
)

TABLE_BEGIN = "!series_matrix_table_begin"
TABLE_END = "!series_matrix_table_end"
PLATFORM_TABLE_BEGIN = "!platform_table_begin"


def _ssl_context() -> ssl.SSLContext:
    """Build a verifying SSL context that honours the OS trust store when possible.

    Networks that terminate TLS at a proxy present a CA that is in the system
    store but not in Python's bundled ``certifi`` bundle, which makes the default
    context reject the NCBI download. ``truststore`` defers to the OS verifier
    and fixes that without ever turning verification off.
    """
    try:
        import truststore
    except ImportError:
        return ssl.create_default_context()
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def _download(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download ``url`` to ``dest``, skipping if it is already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest

    # Download to a temp name first so an interrupted run cannot leave a
    # truncated file that later looks like a valid cached download.
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with (
        urllib.request.urlopen(request, context=_ssl_context()) as response,
        tmp.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle)
    tmp.replace(dest)
    return dest


def download_series_matrix(dest_dir: Path = RAW_DIR, *, force: bool = False) -> Path:
    """Fetch the gzipped series matrix, skipping the download if it is already present."""
    dest = dest_dir / f"{GEO_ACCESSION}_series_matrix.txt.gz"
    return _download(SERIES_MATRIX_URL, dest, force=force)


def download_platform_annotation(dest_dir: Path = RAW_DIR, *, force: bool = False) -> Path:
    """Fetch the GPL10558 probe annotation, which maps probe IDs to gene symbols."""
    dest = dest_dir / f"{PLATFORM}.annot.gz"
    return _download(ANNOTATION_URL, dest, force=force)


def _split_line(line: str) -> list[str]:
    """Split a series-matrix line on tabs and strip GEO's surrounding quotes."""
    return [field.strip().strip('"') for field in line.rstrip("\n").split("\t")]


def _parse_characteristics(values: list[str]) -> list[tuple[str | None, str]]:
    """Split ``"key: value"`` characteristic cells, tolerating cells with no key."""
    parsed: list[tuple[str | None, str]] = []
    for value in values:
        key, sep, rest = value.partition(":")
        if sep:
            parsed.append((key.strip().lower(), rest.strip()))
        else:
            parsed.append((None, value.strip()))
    return parsed


def parse_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a series matrix into ``(expression, sample_metadata)``.

    ``expression`` is probes (rows) by samples (columns) — the orientation GEO
    ships and the one the probe-level filtering steps want. ``sample_metadata``
    is indexed by GSM accession, with each ``!Sample_characteristics_ch1`` key
    promoted to its own column.
    """
    scalar_fields: dict[str, list[str]] = {}
    characteristics: dict[str, dict[int, str]] = defaultdict(dict)
    accessions: list[str] = []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        reached_table = False
        for line in handle:
            if line.startswith(TABLE_BEGIN):
                reached_table = True
                break

            if not line.startswith("!Sample_"):
                continue

            field, *values = _split_line(line)
            field = field.removeprefix("!Sample_")

            if field == "geo_accession":
                accessions = values
            elif field == "characteristics_ch1":
                # Keys can appear in a different order per sample, so bind each
                # value to its own key rather than to the row it arrived on.
                for accession_idx, (key, value) in enumerate(_parse_characteristics(values)):
                    if key is None:
                        continue
                    characteristics[key][accession_idx] = value
            else:
                scalar_fields.setdefault(field, values)

        if not accessions:
            raise ValueError(f"No !Sample_geo_accession line found in {path}")
        if not reached_table:
            raise ValueError(f"No expression table found in {path}")

        # Hand the still-open handle to pandas, which now sits on the table
        # header. Parsing ~47k x 336 values as Python strings first would cost
        # several GB; read_csv builds the typed columns directly. `comment="!"`
        # blanks the trailing !series_matrix_table_end line so it is skipped.
        expression = pd.read_csv(
            handle,
            sep="\t",
            index_col=0,
            comment="!",
            skip_blank_lines=True,
            na_values=["", "null", "NA"],
        )

    expression.index = expression.index.astype(str)
    expression.index.name = "probe_id"
    expression.columns = [str(column).strip().strip('"') for column in expression.columns]
    expression = expression.astype("float32")

    metadata = pd.DataFrame(index=pd.Index(accessions, name="geo_accession"))
    for field, values in scalar_fields.items():
        if len(values) == len(accessions):
            metadata[field] = values
    for key, by_index in characteristics.items():
        column = [by_index.get(i) for i in range(len(accessions))]
        metadata[_safe_column(key)] = column

    # Keep the two frames aligned and in the same sample order.
    shared = [a for a in accessions if a in expression.columns]
    return expression.loc[:, shared], metadata.loc[shared]


def _safe_column(key: str) -> str:
    """Normalise a GEO characteristic key into a usable column name."""
    return re.sub(r"[^0-9a-z]+", "_", key.lower()).strip("_")


def load_series_matrix(
    dest_dir: Path = RAW_DIR, *, force: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (if needed) and parse GSE138458."""
    return parse_series_matrix(download_series_matrix(dest_dir, force=force))


def parse_platform_annotation(path: Path) -> pd.DataFrame:
    """Parse a GEO ``.annot`` platform file into a probe-indexed annotation frame.

    Only the identity columns are kept — the file also carries GO terms and the
    full probe sequence, which we do not need and which dominate its size.
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(PLATFORM_TABLE_BEGIN):
                break
        else:
            raise ValueError(f"No platform table found in {path}")

        annotation = pd.read_csv(
            handle,
            sep="\t",
            index_col=0,
            comment="!",
            skip_blank_lines=True,
            usecols=["ID", "Gene symbol", "Gene title", "Gene ID"],
            dtype=str,
        )

    annotation.index = annotation.index.astype(str)
    annotation.index.name = "probe_id"
    return annotation.rename(
        columns={"Gene symbol": "gene_symbol", "Gene title": "gene_title", "Gene ID": "gene_id"}
    )


def load_probe_annotation(dest_dir: Path = RAW_DIR, *, force: bool = False) -> pd.DataFrame:
    """Download (if needed) and parse the GPL10558 probe-to-gene annotation."""
    return parse_platform_annotation(download_platform_annotation(dest_dir, force=force))
