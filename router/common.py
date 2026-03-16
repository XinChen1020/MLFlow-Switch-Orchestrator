"""Shared router utilities for client access, persisted state, and health checks."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict

import httpx
import docker
import mlflow
from mlflow.tracking import MlflowClient

import config as cfg

# ---- Clients ----
mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
ml_client = MlflowClient()


class _LazyDockerClient:
    """
    Lazily construct the Docker client on first use.

    This keeps module imports cheap for unit tests. Without this wrapper,
    importing modules that reference `docker_client` tries to resolve the
    configured Docker host immediately, which fails outside the full stack.
    """

    def __init__(self, base_url: str):
        self._base_url = base_url
        self._client = None

    def _get_client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.DockerClient(base_url=self._base_url)
        return self._client

    def __getattr__(self, name: str):
        return getattr(self._get_client(), name)


docker_client = _LazyDockerClient(base_url=cfg.DOCKER_HOST)

# ---- State ----
DEFAULT_STATE = {
    "active": None,
    "url": None,
    "public_url": None,
    "model_uri": None,
    "model_version": None,
    "model_alias": None,
    "ts": None,
}


def load_state() -> Dict[str, Any]:
    """Load the active deployment state from disk or return the empty default."""
    try:
        with open(cfg.STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return dict(DEFAULT_STATE)


def save_state(s: Dict[str, Any]) -> None:
    """Persist deployment state atomically so restarts see a consistent file."""
    os.makedirs(os.path.dirname(cfg.STATE_PATH), exist_ok=True)
    tmp = cfg.STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, cfg.STATE_PATH)


# ---- Small shared utils ----

def unique(prefix: str) -> str:
    """Generate a readable, low-collision name for short-lived containers."""
    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"


def ping(url: str, timeout: float = 3.0) -> bool:
    """Check the conventional `/ping` endpoint used by the serving runtime."""
    if not url:
        return False

    try:
        r = httpx.get(url.rstrip("/") + "/ping", timeout=timeout)
        return (200 <= r.status_code < 300) or (r.text.strip().upper() == "OK")
    except Exception:
        return False
