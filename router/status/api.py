from __future__ import annotations
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

import config as cfg
from common import load_state, ping

router = APIRouter()

class StatusResp(BaseModel):
    active: Optional[str] = None
    url: Optional[str] = None
    public_url: Optional[str] = None
    healthy: bool = False
    model_name: Optional[str] = None
    model_uri: Optional[str] = None
    model_version: Optional[int] = None
    model_alias: Optional[str] = None
    ts: Optional[float] = None


def _model_name_from_uri(model_uri: Optional[str]) -> Optional[str]:
    if not model_uri or not model_uri.startswith("models:/"):
        return None

    parts = model_uri.split("/")
    if len(parts) < 3:
        return None
    return parts[1].removeprefix("models:")


@router.get("/status", response_model=StatusResp)
def status() -> StatusResp:
    s = load_state()
    internal = s.get("url")
    public_url = s.get("public_url") or f"http://{cfg.PUBLIC_HOST}:{cfg.PROXY_PUBLIC_PORT}"
    model_uri = s.get("model_uri")
    return StatusResp(
        active=s.get("active"),
        url=internal,
        public_url=public_url,
        healthy=ping(internal) if internal else False,
        model_name=_model_name_from_uri(model_uri),
        model_uri=model_uri,
        model_version=s.get("model_version"),
        model_alias=s.get("model_alias"),
        ts=s.get("ts"),
    )
