"""
Focused unit tests for deployment status reporting.

These tests stay fast by avoiding Docker and MLflow. They validate the small
pieces of logic that shape the `/status` response shown in demos and reviews.
"""

from __future__ import annotations

import status.api as status_api


def test_model_name_from_uri_parses_mlflow_model_uris() -> None:
    """Model names should be extracted only from valid MLflow registry URIs."""
    assert status_api.model_name_from_uri("models:/DiabetesRF/7") == "DiabetesRF"
    assert status_api.model_name_from_uri("models:/MyModel/production") == "MyModel"
    assert status_api.model_name_from_uri(None) is None
    assert status_api.model_name_from_uri("runs:/abc123/model") is None


def test_status_returns_model_metadata_and_public_url_fallback(monkeypatch) -> None:
    """
    The status endpoint should surface both model metadata and health state.

    `monkeypatch` is used here to replace external dependencies so this stays a
    unit test:
    - `load_state()` returns a fake persisted deployment state
    - `ping()` avoids making a real network request
    - config values are overridden so the public URL is deterministic
    """
    state = {
        "active": "serve-cand-123",
        "url": "http://serve-cand-123:8080",
        "public_url": None,
        "model_name": None,
        "model_uri": "models:/DiabetesRF/7",
        "model_version": 7,
        "model_alias": "deployed",
        "serve_image": "server-sklearn-model-1:latest",
        "previous": None,
        "ts": 1735689600.0,
    }

    # Replace the persisted state loader with a fixed in-memory test payload.
    monkeypatch.setattr(status_api, "load_state", lambda: state)
    # Pretend the active deployment is healthy without doing a real HTTP ping.
    monkeypatch.setattr(status_api, "ping", lambda url: url == state["url"])
    # Override config so the public URL fallback is predictable in assertions.
    monkeypatch.setattr(status_api.cfg, "PUBLIC_HOST", "demo-host")
    monkeypatch.setattr(status_api.cfg, "PROXY_PUBLIC_PORT", 9100)

    response = status_api.status()

    assert response.active == "serve-cand-123"
    assert response.url == "http://serve-cand-123:8080"
    assert response.public_url == "http://demo-host:9100"
    assert response.healthy is True
    assert response.model_name == "DiabetesRF"
    assert response.model_uri == "models:/DiabetesRF/7"
    assert response.model_version == 7
    assert response.model_alias == "deployed"
    assert response.ts == 1735689600.0
