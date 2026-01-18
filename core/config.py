"""
Config loader for PaperKG.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


class Config:
    """Load YAML config with env overlay."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        load_dotenv()
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Config not found: {self.path}")
        with self.path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def llm(self) -> Dict[str, Any]:
        return self.data.get('llm', {})

    @property
    def providers(self) -> Dict[str, Any]:
        return self.llm.get('providers', {})

    @property
    def agent_configs(self) -> Dict[str, Any]:
        return self.llm.get('agents', {})

    @property
    def default_provider(self) -> str:
        return self.llm.get('default_provider', 'deepseek')

    def resolve_api_key(self, provider_name: str) -> str | None:
        provider = self.providers.get(provider_name, {})
        env_key = provider.get('api_key_env')
        if env_key:
            return os.getenv(env_key)
        return None
