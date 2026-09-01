"""Download GSE138458 from GEO and report what arrived.

Usage:
    python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse

from biomedical_ml.config import RAW_DIR, ensure_dirs
from biomedical_ml.data import GEO_ACCESSION, download_series_matrix, parse_series_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Download {GEO_ACCESSION} from NCBI GEO.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file is already cached."
    )
    args = parser.parse_args()

    ensure_dirs()
    path = download_series_matrix(RAW_DIR, force=args.force)
    print(f"Series matrix: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    expression, metadata = parse_series_matrix(path)
    print(f"Expression:    {expression.shape[0]} probes x {expression.shape[1]} samples")
    print(f"Metadata:      {metadata.shape[1]} fields")
    print("\nCharacteristic fields:")
    for column in metadata.columns:
        n_unique = metadata[column].nunique(dropna=True)
        print(f"  {column:<28} {n_unique:>4} unique")


if __name__ == "__main__":
    main()
