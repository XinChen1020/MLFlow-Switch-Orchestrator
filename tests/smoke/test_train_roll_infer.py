#!/usr/bin/env python3
"""
Smoke test for the default train-roll-infer happy path.

This script intentionally covers only the primary demo flow:
1. trigger train-and-roll through the router
2. wait for /status to report a healthy active deployment
3. call the stable inference endpoint
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # pragma: no cover - optional during direct script execution
    pytest = None

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _smoke_common import (
    DEFAULT_INFERENCE_PAYLOAD,
    assert_inference_response,
    assert_status_matches_train,
    assert_train_response,
    default_train_parameters,
    json_request,
    load_config,
    status_summary,
    train_summary,
    trigger_train_then_roll,
    wait_for_status,
)


def run_smoke_test() -> None:
    """Run the end-to-end happy path against a live local stack."""
    config = load_config()

    print(f"Triggering train-and-roll for {config.trainer_name}...", flush=True)
    train_response = trigger_train_then_roll(
        config,
        parameters=default_train_parameters(config.trainer_name),
    )
    assert_train_response(train_response)
    print(json.dumps(train_summary(train_response), indent=2), flush=True)

    print("Waiting for /status to report a healthy deployment...", flush=True)
    status_response = wait_for_status(config)
    assert_status_matches_train(status_response, train_response)
    print(json.dumps(status_summary(status_response), indent=2), flush=True)

    print("Calling inference endpoint...", flush=True)
    inference_response = json_request(
        "POST",
        f"{config.inference_url}/invocations",
        payload=DEFAULT_INFERENCE_PAYLOAD,
        timeout=config.request_timeout_seconds,
    )
    assert_inference_response(inference_response)
    print(json.dumps(inference_response, indent=2), flush=True)

    print("Smoke test completed successfully.", flush=True)


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
