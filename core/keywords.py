"""Keyword extraction utilities."""
from __future__ import annotations

import re
from typing import List, Tuple


_KEYWORD_PATTERNS = [
    re.compile(r"^\s*keywords?\s*[:：]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\s*index\s*terms?\s*[:：]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\s*key\s*words?\s*[:：]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\s*关键词\s*[:：]\s*(.+)$", re.IGNORECASE),
]

_SPLIT_PATTERN = re.compile(r"[;,，；、]\s*")


def _clean_keyword(token: str) -> str:
    token = token.strip().strip(".。;；,，")
    token = re.sub(r"\s+", " ", token)
    return token


def _filter_keyword(token: str) -> bool:
    if not token:
        return False
    if len(token) < 2:
        return False
    if any(sym in token for sym in ["$", "\\", "="]):
        return False
    return True


def extract_explicit_keywords(text: str) -> List[str]:
    keywords: List[str] = []
    for line in text.splitlines():
        for pattern in _KEYWORD_PATTERNS:
            match = pattern.match(line)
            if match:
                chunk = match.group(1)
                parts = _SPLIT_PATTERN.split(chunk)
                for part in parts:
                    kw = _clean_keyword(part)
                    if _filter_keyword(kw):
                        keywords.append(kw)
    return _dedupe_keywords(keywords)


def extract_yake_keywords(text: str, max_keywords: int = 12, language: str = "en") -> List[str]:
    try:
        import yake
    except Exception:
        return []

    # YAKE expects reasonably sized text; keep it bounded to avoid noise
    text = text[:200000]
    extractor = yake.KeywordExtractor(lan=language, n=3, top=max_keywords)
    raw = extractor.extract_keywords(text)
    keywords = []
    for kw, _score in raw:
        kw = _clean_keyword(kw)
        if _filter_keyword(kw):
            keywords.append(kw)
    return _dedupe_keywords(keywords)


def extract_keywords(text: str, max_keywords: int = 12, language: str = "en") -> List[str]:
    explicit = extract_explicit_keywords(text)
    if explicit:
        return explicit
    return extract_yake_keywords(text, max_keywords=max_keywords, language=language)


def _dedupe_keywords(keywords: List[str]) -> List[str]:
    seen = set()
    result = []
    for kw in keywords:
        norm = kw.lower().strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(kw)
    return result
