#!/usr/bin/env python3
"""
Generate the checked-in demo dataset used by the reference training backends.

The project keeps a small diabetes regression CSV in `demo_data/` so the
training containers have a deterministic default dataset without generating one
at runtime. Re-run this script if you want to refresh that CSV from
`sklearn.datasets.load_diabetes`.
"""

from __future__ import annotations

from pathlib import Path

try:
    from sklearn.datasets import load_diabetes
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required to regenerate demo_data/diabetes.csv. "
        "Run this script from a project environment such as "
        "`cd model-images/sklearn-model-1 && uv run python ../../demo_data/prepare_demo_data.py`."
    ) from exc


OUTPUT_PATH = Path(__file__).with_name("diabetes.csv")


def main() -> None:
    """Write the diabetes dataset to `demo_data/diabetes.csv`."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_diabetes(as_frame=True).frame
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
