"""LLM client and providers."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional

import aiohttp


class LlmClient:
    def __init__(self, base_url: str, api_key: str, headers: Optional[Dict[str, str]] = None, timeout: int = 300):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = headers or {}
        self.timeout = timeout

    async def call(self, model: str, prompt: str, max_tokens: int, temperature: float, retries: int = 3, retry_delay: int = 2) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    **self.headers,
                }
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.base_url, headers=headers, json=data) as resp:
                        resp.raise_for_status()
                        payload = await resp.json()
                        return payload["choices"][0]["message"]["content"]
            except Exception as exc:
                last_exc = exc
                if attempt < retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("LLM call failed")
