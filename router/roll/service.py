"""Rollout service for promoting MLflow model versions behind a stable endpoint."""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import httpx
from docker.types import Healthcheck
from fastapi import HTTPException

import config as cfg
from common import docker_client, load_state, ml_client, model_name_from_uri, ping, save_state, unique

logger = logging.getLogger(__name__)


class RollService:
    def __init__(self, *, proxy_admin_url: str | None = None):
        self._ml = ml_client
        self._docker = docker_client
        self._admin_url = (proxy_admin_url or cfg.PROXY_ADMIN_URL).rstrip("/") + "/load"
        self._caddy_template_path = Path(__file__).with_name("caddy.json")

    # --- public API ---
    def roll(
        self,
        *,
        name: str,
        ref: Union[str, int],
        wait_ready_seconds: int,
        serve_image: str | None = None,
    ) -> Dict[str, Any]:
        target_uri, version = self._resolve_models_uri(name, ref)
        image = self._resolve_serve_image(serve_image)

        candidate_name = unique("serve-cand")
        logger.info(
            "Starting candidate serving container %s for %s version %s",
            candidate_name,
            name,
            version,
        )
        self._start_runtime(
            candidate_name,
            cfg.COMPOSE_NETWORK,
            target_uri,
            serve_image=image,
        )
        cand_internal = f"http://{candidate_name}:{cfg.SERVE_PORT}"

        deadline = time.time() + wait_ready_seconds
        while time.time() < deadline:
            if ping(cand_internal):
                break
            time.sleep(0.5)
        else:
            self._retire(state_name=candidate_name)
            raise HTTPException(status_code=503, detail="candidate not healthy")

        logger.info("Switching proxy to candidate container %s", candidate_name)
        self._proxy_point_to(candidate_name)
        time.sleep(cfg.DRAIN_GRACE_SEC)

        state = load_state()
        previous_state = self._snapshot_state(state)
        self._retire(state.get("active"))

        alias_name: str | None = None
        try:
            self._ml.set_registered_model_alias(name, cfg.PRODUCTION_ALIAS, str(version))
            alias_name = cfg.PRODUCTION_ALIAS
        except Exception as exc:
            logger.warning(
                "Promoted live traffic to %s version %s but failed to set alias %s: %s",
                name,
                version,
                cfg.PRODUCTION_ALIAS,
                exc,
            )

        new_state = {
            "active": candidate_name,
            "url": cand_internal,
            "public_url": f"http://{cfg.PUBLIC_HOST}:{cfg.PROXY_PUBLIC_PORT}",
            "model_name": name,
            "model_uri": target_uri,
            "model_version": version,
            "model_alias": alias_name,
            "serve_image": image,
            "previous": previous_state,
            "ts": time.time(),
        }
        save_state(new_state)

        if alias_name:
            logger.info(
                "Promoted %s version %s to alias %s",
                name,
                version,
                alias_name,
            )
        else:
            logger.info("Promoted %s version %s without updating a registry alias", name, version)
        return {
            "active": candidate_name,
            "url": cand_internal,
            "public_url": new_state["public_url"],
            "model_uri": target_uri,
            "alias": alias_name,
            "version": version,
        }

    def rollback(
        self,
        *,
        wait_ready_seconds: int,
        serve_image: str | None = None,
    ) -> Dict[str, Any]:
        """Roll back to the previously active deployment captured in state."""
        state = load_state()
        previous = state.get("previous")
        if not isinstance(previous, dict):
            raise HTTPException(status_code=409, detail="no previous deployment recorded for rollback")

        model_name = previous.get("model_name") or model_name_from_uri(previous.get("model_uri"))
        version = previous.get("model_version")
        if not model_name or version is None:
            raise HTTPException(
                status_code=409,
                detail="previous deployment is missing model_name or model_version",
            )

        rollback_image = serve_image or previous.get("serve_image") or state.get("serve_image")
        if not rollback_image:
            raise HTTPException(
                status_code=409,
                detail="previous deployment is missing serve_image; provide one explicitly",
            )

        logger.info(
            "Rolling back to previous deployment %s version %s using serve_image=%s",
            model_name,
            version,
            rollback_image,
        )
        return self.roll(
            name=model_name,
            ref=int(version),
            wait_ready_seconds=wait_ready_seconds,
            serve_image=rollback_image,
        )

    # --- helpers ---
    def _resolve_models_uri(self, name: str, ref: Union[str, int]) -> Tuple[str, int]:
        try:
            if isinstance(ref, str) and ref.startswith("@"):  # alias
                mv = self._ml.get_model_version_by_alias(name, ref[1:])
                version = int(mv.version)
                return f"models:/{name}/{version}", version
            if isinstance(ref, int):
                version = int(ref)
                return f"models:/{name}/{version}", version
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"model/alias not found: {e}")
        raise HTTPException(status_code=400, detail="ref must be '@alias' or integer version")

    def _start_runtime(
        self,
        name: str,
        network: str,
        model_uri: str,
        *,
        serve_image: str | None = None,
    ) -> str:
        # The serving runtime is considered ready once its health endpoint is responsive.
        healthcheck = Healthcheck(
            test=["CMD", "curl", "--f", f"http://localhost:{cfg.SERVE_PORT}/health"],
            interval=5 * 10**9,  # 5s in nanoseconds
            timeout=5 * 10**9,  # 5s
            start_period=5 * 10**9,  # 5s
            retries=3,
        )

        container = self._docker.containers.run(
            image=serve_image,
            name=name,
            detach=True,
            network=network,
            environment={
                "MLFLOW_TRACKING_URI": cfg.MLFLOW_TRACKING_URI,
                "SERVE_MODEL_URI": model_uri,
                "SERVE_PORT": str(cfg.SERVE_PORT),
            },
            restart_policy={"Name": "on-failure", "MaximumRetryCount": 1},
            labels={"app": "mlflow-serve"},
            healthcheck=healthcheck,
        )
        return container.id

    @staticmethod
    def _resolve_serve_image(serve_image: str | None) -> str:
        """Resolve the serving image from the request or router configuration."""
        image = serve_image or cfg.SERVE_IMAGE
        if not image:
            raise HTTPException(
                status_code=500,
                detail="Serving image not configured; supply serve_image via the request or specs.",
            )
        return image

    @staticmethod
    def _snapshot_state(state: Dict[str, Any]) -> Dict[str, Any] | None:
        """Capture the currently active deployment so it can be rolled back to later."""
        if not state.get("active"):
            return None

        snapshot = {
            "active": state.get("active"),
            "url": state.get("url"),
            "public_url": state.get("public_url"),
            "model_name": state.get("model_name") or model_name_from_uri(state.get("model_uri")),
            "model_uri": state.get("model_uri"),
            "model_version": state.get("model_version"),
            "model_alias": state.get("model_alias"),
            "serve_image": state.get("serve_image"),
            "ts": state.get("ts"),
        }
        return snapshot

    def _retire(self, state_name: str | None) -> None:
        """
        Stop and remove a container by name, ignoring errors.
        """
        if not state_name:
            return
        try:
            c = self._docker.containers.get(state_name)
            c.stop(timeout=5)
            c.remove()
        except Exception as exc:
            logger.warning("Failed to retire container %s: %s", state_name, exc)

    def _make_caddy_config(self, target_container: str) -> Dict[str, Any]:
        """
        Load the saved Caddy config template and patch the runtime target values.
        """
        cfg_json = self._load_caddy_template()
        cfg_json["admin"]["listen"] = ":2019"
        cfg_json["apps"]["http"]["servers"]["srv0"]["listen"] = [f":{cfg.PROXY_PUBLIC_PORT}"]
        cfg_json["apps"]["http"]["servers"]["srv0"]["routes"][0]["handle"][0]["upstreams"] = [
            {"dial": f"{target_container}:{cfg.SERVE_PORT}"}
        ]
        return cfg_json

    def _load_caddy_template(self) -> Dict[str, Any]:
        """Read the checked-in Caddy JSON template and return a mutable copy."""
        try:
            with self._caddy_template_path.open("r", encoding="utf-8") as handle:
                return deepcopy(json.load(handle))
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"failed to load Caddy config template: {exc}",
            ) from exc

    def _proxy_point_to(self, target_container: str) -> None:
        """
        Point Caddy proxy to target container using POST.
        """
        cfg_json = self._make_caddy_config(target_container)
        r = httpx.post(self._admin_url, json=cfg_json, timeout=5.0)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"proxy switch failed: {e.response.text}")
