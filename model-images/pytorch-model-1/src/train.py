#!/usr/bin/env python3
"""
PyTorch regression training with MLflow tracking and registration.

This trainer intentionally mirrors the scikit-learn demo flow so the repo shows
the same router/orchestration path working across two frameworks.

Environment:
  MLFLOW_TRACKING_URI        MLflow tracking server URI
  MLFLOW_EXPERIMENT          Experiment name (default: "diabetes_torch_demo")
  REGISTERED_MODEL_NAME      Model Registry name (default: "DiabetesTorch")
  DATASET_PATH               CSV path (default: "/app/demo_data/diabetes.csv")
  TARGET_COLUMN              Target column name (default: "target")
  DATASET_NAME               Dataset display name logged to MLflow (default: "demo_diabetes")
  DATASET_VERSION            Dataset version metadata logged to MLflow (default: "v1")
  DATASET_SAMPLE_ROWS        Optional row cap for faster demo or smoke-test runs
  TEST_SIZE                  Test fraction (default "0.2")
  VAL_SIZE                   Validation fraction of non-test portion (default "0.2")
  RANDOM_STATE               Random seed (default "42")
  HIDDEN_DIM                 Hidden layer width (default "32")
  LEARNING_RATE              Adam learning rate (default "0.001")
  EPOCHS                     Number of training epochs (default "250")
  BATCH_SIZE                 Batch size (default "32")
  MLFLOW_RUN_ID              If set, reuse an existing run (from orchestrator)
  SET_ALIAS                  Optional alias to attach to the created model version
  LOG_LEVEL                  Logging level (default "INFO")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from sklearn import __version__ as sklearn_version
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from helpers import (
    DEFAULT_DATASET_PATH,
    DEFAULT_TARGET_COLUMN,
    log_dataset_stage,
    load_csv,
    resolve_version_for_run,
    sample_rows,
    split_train_val_test,
)


class NormalizedRegressor(nn.Module):
    """Small MLP regressor with input normalization baked into the model."""

    def __init__(self, input_dim: int, hidden_dim: int, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        safe_std = np.where(std < 1e-6, 1.0, std)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(safe_std, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = (inputs - self.mean) / self.std
        return self.network(normalized)


EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "diabetes_torch_demo")
mlflow.set_experiment(EXPERIMENT)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("trainer")
client = MlflowClient()


def _to_loader(X: pd.DataFrame, y: pd.Series, batch_size: int, shuffle: bool) -> DataLoader:
    features = torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32)
    targets = torch.tensor(y.to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
    return DataLoader(TensorDataset(features, targets), batch_size=batch_size, shuffle=shuffle)


def _evaluate_regression(model: nn.Module, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    features = torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32)
    with torch.no_grad():
        preds = model(features).cpu().numpy().reshape(-1)

    target = y.to_numpy(dtype=np.float32)
    residual = preds - target
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    target_var = float(np.var(target))
    r2 = 0.0 if target_var == 0.0 else float(1.0 - (np.var(residual) / target_var))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    dataset_path = os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH)
    target_col = os.getenv("TARGET_COLUMN", DEFAULT_TARGET_COLUMN)
    dataset_name = os.getenv("DATASET_NAME", "demo_diabetes")
    dataset_version = os.getenv("DATASET_VERSION", "v1")

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    X, y = load_csv(dataset_path, target_col)
    test_size = float(os.getenv("TEST_SIZE", "0.2"))
    val_size = float(os.getenv("VAL_SIZE", "0.2"))
    random_state = int(os.getenv("RANDOM_STATE", "42"))
    dataset_sample_rows = int(os.getenv("DATASET_SAMPLE_ROWS", "0"))
    hidden_dim = int(os.getenv("HIDDEN_DIM", "32"))
    learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))
    epochs = int(os.getenv("EPOCHS", "250"))
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    registered_model = os.getenv("REGISTERED_MODEL_NAME", "DiabetesTorch")
    run_id_env = os.getenv("MLFLOW_RUN_ID")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    original_row_count = len(X)
    X, y = sample_rows(X, y, max_rows=dataset_sample_rows, random_state=random_state)
    X_tr, y_tr, X_va, y_va, X_te, y_te = split_train_val_test(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )
    train_loader = _to_loader(X_tr, y_tr, batch_size=batch_size, shuffle=True)

    mean = X_tr.to_numpy(dtype=np.float32).mean(axis=0)
    std = X_tr.to_numpy(dtype=np.float32).std(axis=0)
    model = NormalizedRegressor(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        mean=mean,
        std=std,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    with (mlflow.start_run(run_id=run_id_env) if run_id_env else mlflow.start_run()) as run:
        run_id = run.info.run_id
        mlflow.set_tags(
            {
                "framework": "pytorch",
                "launch_mode": "orchestrated" if run_id_env else "standalone",
            }
        )

        source_uri = Path(dataset_path).resolve().as_uri()
        log_dataset_stage(X_tr, y_tr, stage="training", name="train_dataset", source=source_uri)
        log_dataset_stage(X_va, y_va, stage="validation", name="val_dataset", source=source_uri)
        log_dataset_stage(X_te, y_te, stage="evaluation", name="test_dataset", source=source_uri)

        mlflow.log_params(
            {
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "random_state": random_state,
                "test_size": test_size,
                "val_size": val_size,
                "feature_count": X.shape[1],
                "sklearn_version": sklearn_version,
                "torch_version": torch.__version__,
                "dataset_path": dataset_path,
                "dataset_name": dataset_name,
                "dataset_version": dataset_version,
                "dataset_row_count": original_row_count,
                "sampled_dataset_row_count": len(X),
                "dataset_sample_rows": dataset_sample_rows,
                "target_column": target_col,
            }
        )

        best_state = None
        best_val_rmse = float("inf")
        for epoch in range(epochs):
            model.train()
            for features, targets in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(features), targets)
                loss.backward()
                optimizer.step()

            model.eval()
            val_metrics = _evaluate_regression(model, X_va, y_va)
            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if epoch in {0, epochs - 1} or (epoch + 1) % max(1, epochs // 5) == 0:
                mlflow.log_metric("validation_rmse", val_metrics["rmse"], step=epoch + 1)

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        scripted_model = torch.jit.script(model)
        input_example = X.head(2).astype(np.float32)
        mlflow.pytorch.log_model(
            pytorch_model=scripted_model,
            name="model",
            input_example=input_example,
            registered_model_name=registered_model,
        )

        # Keep evaluation inputs aligned with the float32 model signature that
        # was logged via mlflow.pytorch.
        eval_df = X_te.astype(np.float32).copy()
        eval_df[target_col] = y_te.to_numpy(dtype=np.float32)
        result = mlflow.evaluate(
            model=f"runs:/{run_id}/model",
            data=eval_df,
            targets=target_col,
            model_type="regressor",
        )

        rmse = result.metrics.get("root_mean_squared_error")
        mae = result.metrics.get("mean_absolute_error")
        r2 = result.metrics.get("r2_score")
        log.info("Evaluation (test): RMSE=%.4f  MAE=%.4f  R2=%.4f", rmse, mae, r2)

        version = resolve_version_for_run(client, registered_model, run_id)
        alias = os.getenv("SET_ALIAS")
        if alias and version is not None:
            client.set_registered_model_alias(registered_model, alias, int(version))
            log.info("Set alias '%s' on %s version %s", alias, registered_model, version)

        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "registered_model": registered_model,
                    "version": version,
                    "alias": alias,
                    "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
