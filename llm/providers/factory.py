"""Provider factory."""
from __future__ import annotations

from typing import Dict

from core.config import Config
from llm.client import LlmClient


def build_client(cfg: Config, provider_name: str) -> LlmClient:
    provider = cfg.providers.get(provider_name, {})
    api_key = cfg.resolve_api_key(provider_name)
    if not api_key:
        raise ValueError(f"Missing API key for provider: {provider_name}")
    base_url = provider.get('base_url')
    if not base_url:
        raise ValueError(f"Missing base_url for provider: {provider_name}")
    headers = provider.get('headers', {})
    return LlmClient(base_url=base_url, api_key=api_key, headers=headers, timeout=cfg.get('workflow', {}).get('api_timeout', 300))
