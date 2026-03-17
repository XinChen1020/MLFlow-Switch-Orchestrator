"""
helpers.py - shared data preparation and MLflow lineage helpers for the PyTorch demo.
"""

from __future__ import annotations

import time
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_DATASET_PATH = "/app/demo_data/diabetes.csv"
DEFAULT_TARGET_COLUMN = "target"


def load_csv(path: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV into numeric features and a target vector."""
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"TARGET_COLUMN '{target_col}' not found in: {path}")

    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    return X, y


def sample_rows(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Optionally down-sample rows for faster demo or smoke-test training."""
    if not max_rows or max_rows <= 0 or len(X) <= max_rows:
        return X, y

    sampled_indices = X.sample(n=max_rows, random_state=random_state).index
    return X.loc[sampled_indices].reset_index(drop=True), y.loc[sampled_indices].reset_index(drop=True)


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split the dataset into train, validation, and test partitions."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_rel = val_size / max(1e-9, (1.0 - test_size))
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, random_state=random_state
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def log_dataset_stage(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    stage: str,
    source: str,
    name: str,
) -> None:
    """Attach a dataset snapshot and a few quality hints to the active MLflow run."""
    df = X.copy()
    df["target"] = y.values if hasattr(y, "values") else y
    ds = mlflow.data.from_pandas(df, source=source, name=name)
    mlflow.log_input(ds, context=stage)

    missing_pct = (df.isnull().to_numpy().sum() / float(df.size)) * 100.0
    mlflow.log_metrics(
        {
            f"{stage}_rows": len(df),
            f"{stage}_columns": len(df.columns),
            f"{stage}_missing_pct": float(missing_pct),
        }
    )


def resolve_version_for_run(
    client,
    model_name: str,
    run_id: str,
    tries: int = 20,
    sleep_s: float = 0.5,
) -> Optional[int]:
    """Look up the MLflow model version created by a specific run."""
    query = f"name = '{model_name}' and run_id = '{run_id}'"
    for _ in range(max(1, tries)):
        mvs = list(client.search_model_versions(query))
        if mvs:
            try:
                return int(mvs[0].version)
            except Exception:
                pass
        time.sleep(sleep_s)
    return None
