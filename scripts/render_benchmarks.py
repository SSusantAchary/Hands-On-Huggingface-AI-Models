"""Placeholder script to turn benchmark CSV rows into SVG charts."""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "benchmarks" / "matrix.csv"
OUTPUT_DIR = ROOT / "assets" / "benchmarks"


def main() -> None:
    if not CSV.exists():
        raise SystemExit("No benchmark data yet. Run a notebook to populate matrix.csv.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    if df.empty:
        raise SystemExit("matrix.csv is empty. Measurements append rows automatically.")
    # TODO: implement rendering logic once measurements are available.
    print("Read", len(df), "rows. Please implement chart rendering and overwrite SVGs.")


if __name__ == "__main__":
    main()
