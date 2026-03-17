#!/usr/bin/env python3
"""
Shared helpers for smoke tests that exercise the live local stack.

These helpers keep the transport, polling, and response validation logic in one
place so the individual smoke scenarios can stay short and readable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request


DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_TRAIN_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_STATUS_ATTEMPTS = 30
DEFAULT_STATUS_SLEEP_SECONDS = 2.0
DEFAULT_WAIT_SECONDS = 600
DEFAULT_TRAINER_NAME = "sklearn-model-1"
DEFAULT_SMOKE_SKLEARN_ESTIMATORS = 32
DEFAULT_SMOKE_SKLEARN_ROLLBACK_ESTIMATORS = 48
DEFAULT_SMOKE_SAMPLE_ROWS = 96
DEFAULT_SMOKE_ROLLBACK_SAMPLE_ROWS = 128
DEFAULT_SMOKE_PYTORCH_EPOCHS = 40
DEFAULT_SMOKE_PYTORCH_ROLLBACK_EPOCHS = 60
DEFAULT_SMOKE_PYTORCH_HIDDEN_DIM = 16

DEFAULT_INFERENCE_PAYLOAD = {
    "dataframe_split": {
        "columns": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
        "data": [[0.03, 1, 0.06, 0.03, 0.04, 0.03, 0.02, 0.03, 0.04, 0.01]],
    }
}


@dataclass
class SmokeConfig:
    router_url: str
    inference_url: str
    trainer_name: str
    wait_seconds: int
    train_request_timeout_seconds: float
    request_timeout_seconds: float
    status_attempts: int
    status_sleep_seconds: float


def load_config() -> SmokeConfig:
    """Load smoke-test runtime settings from the environment."""
    wait_seconds = int(os.getenv("WAIT_SECONDS", str(DEFAULT_WAIT_SECONDS)))
    request_timeout_seconds = float(
        os.getenv("REQUEST_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
    )
    train_request_timeout_seconds = float(
        os.getenv(
            "TRAIN_REQUEST_TIMEOUT_SECONDS",
            str(max(DEFAULT_TRAIN_REQUEST_TIMEOUT_SECONDS, wait_seconds + 60)),
        )
    )
    return SmokeConfig(
        router_url=os.getenv("ROUTER_URL", "http://localhost:8000").rstrip("/"),
        inference_url=os.getenv("INFERENCE_URL", "http://localhost:9000").rstrip("/"),
        trainer_name=os.getenv("TRAINER_NAME", DEFAULT_TRAINER_NAME),
        wait_seconds=wait_seconds,
        train_request_timeout_seconds=train_request_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
        status_attempts=int(os.getenv("STATUS_ATTEMPTS", str(DEFAULT_STATUS_ATTEMPTS))),
        status_sleep_seconds=float(
            os.getenv("STATUS_SLEEP_SECONDS", str(DEFAULT_STATUS_SLEEP_SECONDS))
        ),
    )


def json_request(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> Any:
    """Send a JSON request and return the parsed JSON response."""
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc

    try:
        return json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {url} returned non-JSON output: {raw_body}") from exc


def default_train_parameters(trainer_name: str) -> dict[str, Any]:
    """Use lighter trainer-specific settings so smoke tests finish sooner."""
    if trainer_name == "pytorch-model-1":
        return {
            "DATASET_SAMPLE_ROWS": int(
                os.getenv("SMOKE_DATASET_SAMPLE_ROWS", str(DEFAULT_SMOKE_SAMPLE_ROWS))
            ),
            "EPOCHS": int(os.getenv("SMOKE_PYTORCH_EPOCHS", str(DEFAULT_SMOKE_PYTORCH_EPOCHS))),
            "HIDDEN_DIM": int(
                os.getenv("SMOKE_PYTORCH_HIDDEN_DIM", str(DEFAULT_SMOKE_PYTORCH_HIDDEN_DIM))
            ),
        }
    return {
        "DATASET_SAMPLE_ROWS": int(
            os.getenv("SMOKE_DATASET_SAMPLE_ROWS", str(DEFAULT_SMOKE_SAMPLE_ROWS))
        ),
        "N_ESTIMATORS": int(
            os.getenv("SMOKE_SKLEARN_ESTIMATORS", str(DEFAULT_SMOKE_SKLEARN_ESTIMATORS))
        )
    }


def rollback_train_parameters(trainer_name: str) -> dict[str, Any]:
    """Use a second lightweight override to force a new model version before rollback."""
    if trainer_name == "pytorch-model-1":
        return {
            "DATASET_SAMPLE_ROWS": int(
                os.getenv(
                    "SMOKE_ROLLBACK_DATASET_SAMPLE_ROWS",
                    str(DEFAULT_SMOKE_ROLLBACK_SAMPLE_ROWS),
                )
            ),
            "EPOCHS": int(
                os.getenv(
                    "SMOKE_PYTORCH_ROLLBACK_EPOCHS",
                    str(DEFAULT_SMOKE_PYTORCH_ROLLBACK_EPOCHS),
                )
            ),
            "HIDDEN_DIM": int(
                os.getenv("SMOKE_PYTORCH_HIDDEN_DIM", str(DEFAULT_SMOKE_PYTORCH_HIDDEN_DIM))
            ),
        }
    return {
        "DATASET_SAMPLE_ROWS": int(
            os.getenv(
                "SMOKE_ROLLBACK_DATASET_SAMPLE_ROWS",
                str(DEFAULT_SMOKE_ROLLBACK_SAMPLE_ROWS),
            )
        ),
        "N_ESTIMATORS": int(
            os.getenv(
                "SMOKE_SKLEARN_ROLLBACK_ESTIMATORS",
                str(DEFAULT_SMOKE_SKLEARN_ROLLBACK_ESTIMATORS),
            )
        )
    }


def trigger_train_then_roll(
    config: SmokeConfig,
    *,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Trigger the synchronous train-and-roll route with caller-provided overrides."""
    return json_request(
        "POST",
        f"{config.router_url}/admin/train_then_roll/{config.trainer_name}",
        payload={
            "wait_seconds": config.wait_seconds,
            "parameters": parameters,
        },
        timeout=config.train_request_timeout_seconds,
    )


def wait_for_status(
    config: SmokeConfig,
    *,
    expected_version: int | None = None,
) -> dict[str, Any]:
    """Poll /status until the active deployment is healthy and versioned."""
    last_status: dict[str, Any] | None = None
    for _ in range(config.status_attempts):
        status_response = json_request(
            "GET",
            f"{config.router_url}/status",
            timeout=config.request_timeout_seconds,
        )
        last_status = status_response
        version = status_response.get("model_version")
        if (
            status_response.get("healthy") is True
            and version
            and (expected_version is None or version == expected_version)
        ):
            return status_response
        time.sleep(config.status_sleep_seconds)

    raise AssertionError(
        "Timed out waiting for /status to report a healthy deployment: "
        f"{json.dumps(last_status, indent=2)}"
    )


def assert_train_response(payload: dict[str, Any]) -> None:
    """Validate the minimum fields needed for follow-on smoke steps."""
    required = ["rolled", "run_id", "registered_model", "version", "public_url"]
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise AssertionError(f"train_then_roll response missing fields: {missing}")
    if payload.get("rolled") is not True:
        raise AssertionError(f"train_then_roll did not report a successful rollout: {payload}")


def assert_status_matches_train(
    status_response: dict[str, Any],
    train_response: dict[str, Any],
) -> None:
    """Confirm /status reflects the version produced by train_then_roll."""
    if status_response.get("model_version") != train_response.get("version"):
        raise AssertionError(
            "status model_version does not match the version returned by train_then_roll"
        )
    if status_response.get("healthy") is not True:
        raise AssertionError(f"deployment is not healthy: {status_response}")


def assert_rollback_matches_initial(
    status_response: dict[str, Any],
    initial_train_response: dict[str, Any],
) -> None:
    """Confirm /status returns to the version that was active before rollback."""
    if status_response.get("model_version") != initial_train_response.get("version"):
        raise AssertionError("rollback did not restore the initial model version")
    if status_response.get("healthy") is not True:
        raise AssertionError(f"rolled back deployment is not healthy: {status_response}")


def assert_inference_response(payload: Any) -> None:
    """Accept either the list or dict formats returned by MLflow serving."""
    if isinstance(payload, dict):
        if not payload:
            raise AssertionError("inference response was empty")
        return
    if isinstance(payload, list):
        if not payload:
            raise AssertionError("inference response list was empty")
        return
    raise AssertionError(f"unexpected inference response type: {type(payload).__name__}")


def train_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim the train response to the fields most useful in demo output."""
    return {
        "rolled": payload["rolled"],
        "run_id": payload["run_id"],
        "registered_model": payload["registered_model"],
        "version": payload["version"],
        "public_url": payload["public_url"],
    }


def status_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim the status response to the active deployment summary."""
    return {
        "active": payload.get("active"),
        "healthy": payload.get("healthy"),
        "model_name": payload.get("model_name"),
        "model_version": payload.get("model_version"),
        "model_alias": payload.get("model_alias"),
        "public_url": payload.get("public_url"),
    }
