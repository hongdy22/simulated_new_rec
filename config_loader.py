from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key: str
    default_model: str
    timeout_seconds: float
    endpoints: Dict[str, str]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_api_config(config_path: Optional[Path] = None) -> ApiConfig:
    """
    Load API configuration from local `config.json`, but allow env overrides:
    - OPENAI_BASE_URL
    - OPENAI_API_KEY
    - OPENAI_DEFAULT_MODEL
    """

    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {config_path}. Create it from config.example.json."
        )

    data = _read_json(config_path)
    api = data.get("api") or {}

    env_base_url = os.getenv("OPENAI_BASE_URL")
    env_api_key = os.getenv("OPENAI_API_KEY")
    env_default_model = os.getenv("OPENAI_DEFAULT_MODEL")

    base_url = str(env_base_url or api.get("base_url") or "").strip()
    api_key = str(env_api_key or api.get("api_key") or "").strip()
    default_model = str(env_default_model or api.get("default_model") or "").strip()
    timeout_seconds = float(api.get("timeout_seconds") or 120)
    endpoints = api.get("endpoints") or {}

    if not base_url:
        raise ValueError("API config missing `api.base_url` (or OPENAI_BASE_URL).")
    if not api_key or api_key == "PUT_YOUR_API_KEY_HERE":
        raise ValueError(
            "API config missing `api.api_key` (or OPENAI_API_KEY). "
            "Please set it in config.json."
        )
    if not default_model:
        raise ValueError("API config missing `api.default_model` (or OPENAI_DEFAULT_MODEL).")

    endpoints = {str(k): str(v) for k, v in endpoints.items()}
    return ApiConfig(
        base_url=base_url,
        api_key=api_key,
        default_model=default_model,
        timeout_seconds=timeout_seconds,
        endpoints=endpoints,
    )

