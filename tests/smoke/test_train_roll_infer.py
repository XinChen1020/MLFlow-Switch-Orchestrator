#!/usr/bin/env python3
"""
Smoke test for the demo deployment flow.

This script exercises the happy path against a running local stack:
1. trigger train-and-roll through the router
2. wait for /status to report a healthy active deployment
3. call the stable inference endpoint

It can be run directly as a script or collected by pytest.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request

try:
    import pytest
except ImportError:  # pragma: no cover - optional during direct script execution
    pytest = None


DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_STATUS_ATTEMPTS = 30
DEFAULT_STATUS_SLEEP_SECONDS = 2.0
DEFAULT_WAIT_SECONDS = 600
DEFAULT_TRAINER_NAME = "sklearn-model-1"

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
    request_timeout_seconds: float
    status_attempts: int
    status_sleep_seconds: float


def _load_config() -> SmokeConfig:
    """Load runtime configuration from the environment with local defaults."""
    return SmokeConfig(
        router_url=os.getenv("ROUTER_URL", "http://localhost:8000").rstrip("/"),
        inference_url=os.getenv("INFERENCE_URL", "http://localhost:9000").rstrip("/"),
        trainer_name=os.getenv("TRAINER_NAME", DEFAULT_TRAINER_NAME),
        wait_seconds=int(os.getenv("WAIT_SECONDS", str(DEFAULT_WAIT_SECONDS))),
        request_timeout_seconds=float(
            os.getenv("REQUEST_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
        ),
        status_attempts=int(os.getenv("STATUS_ATTEMPTS", str(DEFAULT_STATUS_ATTEMPTS))),
        status_sleep_seconds=float(
            os.getenv("STATUS_SLEEP_SECONDS", str(DEFAULT_STATUS_SLEEP_SECONDS))
        ),
    )


def _json_request(
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


def run_smoke_test() -> None:
    """Run the end-to-end happy path against a live local stack."""
    config = _load_config()

    print(f"Triggering train-and-roll for {config.trainer_name}...", flush=True)
    train_response = _json_request(
        "POST",
        f"{config.router_url}/admin/train_then_roll/{config.trainer_name}",
        payload={
            "wait_seconds": config.wait_seconds,
            "parameters": {"N_ESTIMATORS": 256},
        },
        timeout=config.request_timeout_seconds,
    )
    _assert_train_response(train_response)
    print(json.dumps(_train_summary(train_response), indent=2), flush=True)

    print("Waiting for /status to report a healthy deployment...", flush=True)
    status_response = _wait_for_status(config)
    _assert_status_matches_train(status_response, train_response)
    print(json.dumps(_status_summary(status_response), indent=2), flush=True)

    print("Calling inference endpoint...", flush=True)
    inference_response = _json_request(
        "POST",
        f"{config.inference_url}/invocations",
        payload=DEFAULT_INFERENCE_PAYLOAD,
        timeout=config.request_timeout_seconds,
    )
    _assert_inference_response(inference_response)
    print(json.dumps(inference_response, indent=2), flush=True)

    print("Smoke test completed successfully.", flush=True)


def _assert_train_response(payload: dict[str, Any]) -> None:
    """Validate the minimum fields needed for the rest of the smoke test."""
    required = ["rolled", "run_id", "registered_model", "version", "public_url"]
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise AssertionError(f"train_then_roll response missing fields: {missing}")
    if payload.get("rolled") is not True:
        raise AssertionError(f"train_then_roll did not report a successful rollout: {payload}")


def _wait_for_status(config: SmokeConfig) -> dict[str, Any]:
    """Poll /status until the active deployment is healthy and versioned."""
    last_status: dict[str, Any] | None = None
    for _ in range(config.status_attempts):
        status_response = _json_request(
            "GET",
            f"{config.router_url}/status",
            timeout=config.request_timeout_seconds,
        )
        last_status = status_response
        # The rollout is considered ready only once health checks pass and the
        # active model version is visible in the persisted deployment state.
        if status_response.get("healthy") is True and status_response.get("model_version"):
            return status_response
        time.sleep(config.status_sleep_seconds)

    raise AssertionError(
        "Timed out waiting for /status to report a healthy deployment: "
        f"{json.dumps(last_status, indent=2)}"
    )


def _assert_status_matches_train(
    status_response: dict[str, Any], train_response: dict[str, Any]
) -> None:
    """Confirm /status reflects the version produced by train_then_roll."""
    if status_response.get("model_version") != train_response.get("version"):
        raise AssertionError(
            "status model_version does not match the version returned by train_then_roll"
        )
    if status_response.get("healthy") is not True:
        raise AssertionError(f"deployment is not healthy: {status_response}")


def _assert_inference_response(payload: Any) -> None:
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


def _train_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim the train response to the fields most useful in demo output."""
    return {
        "rolled": payload["rolled"],
        "run_id": payload["run_id"],
        "registered_model": payload["registered_model"],
        "version": payload["version"],
        "public_url": payload["public_url"],
    }


def _status_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim the status response to the active deployment summary."""
    return {
        "active": payload.get("active"),
        "healthy": payload.get("healthy"),
        "model_name": payload.get("model_name"),
        "model_version": payload.get("model_version"),
        "model_alias": payload.get("model_alias"),
        "public_url": payload.get("public_url"),
    }


def test_train_roll_infer_smoke() -> None:
    # Keep smoke tests opt-in under pytest so normal local test runs do not
    # unexpectedly depend on Docker services already being up.
    if pytest is None:
        run_smoke_test()
        return

    if os.getenv("RUN_SMOKE_TESTS") != "1":
        pytest.skip("Set RUN_SMOKE_TESTS=1 to run smoke tests against a live stack.")
    run_smoke_test()


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
