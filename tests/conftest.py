"""Shared fixtures.

Unit tests run on synthetic frames so the suite works offline. The few tests
that need the real 81 MB GEO download are marked ``integration`` and skip
automatically when it has not been fetched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomedical_ml.config import RAW_DIR
from biomedical_ml.data import GEO_ACCESSION


@pytest.fixture
def synthetic_expression() -> pd.DataFrame:
    """Probes-by-samples expression with two all-NaN placeholder columns."""
    rng = np.random.default_rng(0)
    samples = [f"GSM{i}" for i in range(10)]
    probes = [f"ILMN_{i}" for i in range(6)]
    frame = pd.DataFrame(
        rng.normal(8.0, 1.0, size=(len(probes), len(samples))).astype("float32"),
        index=pd.Index(probes, name="probe_id"),
        columns=samples,
    )
    frame[["GSM8", "GSM9"]] = np.nan
    return frame


@pytest.fixture
def synthetic_metadata() -> pd.DataFrame:
    """Sample table mirroring the GEO fields the pipeline relies on."""
    return pd.DataFrame(
        {
            # Subjects 0 and 1 each appear twice — the repeated-measures case.
            "subject_id": ["0", "0", "1", "1", "2", "3", "4", "5", "6", "7"],
            "case_control": ["SLE Case"] * 6 + ["Control"] * 4,
            "description": [f"20031968011{i // 4}_{chr(65 + i % 4)}" for i in range(10)],
        },
        index=pd.Index([f"GSM{i}" for i in range(10)], name="geo_accession"),
    )


def _real_data_available() -> bool:
    return (RAW_DIR / f"{GEO_ACCESSION}_series_matrix.txt.gz").exists()


requires_real_data = pytest.mark.skipif(
    not _real_data_available(),
    reason="GEO download missing; run scripts/download_data.py",
)