#!/usr/bin/env python3
"""
Smoke test for the explicit rollback path.

This scenario assumes a live local stack and validates:
1. an initial train-and-roll
2. a second promotion that creates a new model version
3. an explicit `/admin/rollback`
4. `/status` returning to the initial version
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
    assert_rollback_matches_initial,
    assert_train_response,
    default_train_parameters,
    json_request,
    load_config,
    rollback_train_parameters,
    status_summary,
    train_summary,
    trigger_train_then_roll,
    wait_for_status,
)


def run_rollback_smoke_test() -> None:
    """Run the rollback smoke scenario against a live local stack."""
    config = load_config()

    print(f"Triggering initial train-and-roll for {config.trainer_name}...", flush=True)
    initial_train_response = trigger_train_then_roll(
        config,
        parameters=default_train_parameters(config.trainer_name),
    )
    assert_train_response(initial_train_response)
    print(json.dumps(train_summary(initial_train_response), indent=2), flush=True)

    print("Triggering second train-and-roll for rollback setup...", flush=True)
    second_train_response = trigger_train_then_roll(
        config,
        parameters=rollback_train_parameters(config.trainer_name),
    )
    assert_train_response(second_train_response)
    if second_train_response.get("version") == initial_train_response.get("version"):
        raise AssertionError("second train_then_roll did not produce a new model version")
    print(json.dumps(train_summary(second_train_response), indent=2), flush=True)

    print("Calling rollback endpoint...", flush=True)
    rollback_response = json_request(
        "POST",
        f"{config.router_url}/admin/rollback",
        payload={"wait_ready_seconds": config.wait_seconds},
        timeout=config.train_request_timeout_seconds,
    )
    print(json.dumps(rollback_response, indent=2), flush=True)

    print("Waiting for /status to reflect the rollback...", flush=True)
    rolled_back_status = wait_for_status(
        config,
        expected_version=initial_train_response.get("version"),
    )
    assert_rollback_matches_initial(rolled_back_status, initial_train_response)
    print(json.dumps(status_summary(rolled_back_status), indent=2), flush=True)

    print("Rollback smoke test completed successfully.", flush=True)


def test_rollback_smoke() -> None:
    # Keep smoke tests opt-in under pytest so normal local test runs do not
    # unexpectedly depend on Docker services already being up.
    if pytest is None:
        run_rollback_smoke_test()
        return

    if os.getenv("RUN_SMOKE_TESTS") != "1":
        pytest.skip("Set RUN_SMOKE_TESTS=1 to run smoke tests against a live stack.")
    run_rollback_smoke_test()


if __name__ == "__main__":
    try:
        run_rollback_smoke_test()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
