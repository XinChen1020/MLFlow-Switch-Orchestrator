"""
Focused unit tests for rollback-related rollout state behavior.

These tests validate the minimal state transitions needed for explicit rollback
without requiring Docker, MLflow, or a running stack.
"""

from __future__ import annotations

import pytest

from fastapi import HTTPException

import roll.service as roll_service


def test_snapshot_state_captures_previous_deployment_metadata() -> None:
    """The previous deployment snapshot should keep the fields needed for rollback."""
    state = {
        "active": "serve-cand-123",
        "url": "http://serve-cand-123:8080",
        "public_url": "http://localhost:9000",
        "model_name": None,
        "model_uri": "models:/DiabetesRF/7",
        "model_version": 7,
        "model_alias": "deployed",
        "serve_image": "server-sklearn-model-1:latest",
        "ts": 1735689600.0,
    }

    snapshot = roll_service.RollService._snapshot_state(state)

    assert snapshot == {
        "active": "serve-cand-123",
        "url": "http://serve-cand-123:8080",
        "public_url": "http://localhost:9000",
        "model_name": "DiabetesRF",
        "model_uri": "models:/DiabetesRF/7",
        "model_version": 7,
        "model_alias": "deployed",
        "serve_image": "server-sklearn-model-1:latest",
        "ts": 1735689600.0,
    }


def test_rollback_requires_a_previous_deployment(monkeypatch) -> None:
    """Rollback should fail clearly when there is no stored deployment to restore."""
    monkeypatch.setattr(roll_service, "load_state", lambda: {"previous": None})

    service = roll_service.RollService()
    with pytest.raises(HTTPException) as exc_info:
        service.rollback(wait_ready_seconds=30)

    assert exc_info.value.status_code == 409
    assert "no previous deployment recorded" in str(exc_info.value.detail)
