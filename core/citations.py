"""Citation parsing utilities."""
from __future__ import annotations

import re
from typing import List


def _expand_numeric_tokens(token: str) -> List[str]:
    token = token.strip()
    if not token:
        return []
    # Normalize separators
    token = token.replace("\u2013", "-").replace("\u2014", "-").replace("~", "-")
    if "-" in token:
        parts = [p.strip() for p in token.split("-") if p.strip()]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            start = int(parts[0])
            end = int(parts[1])
            if start <= end:
                return [str(i) for i in range(start, end + 1)]
            return [str(i) for i in range(start, end - 1, -1)]
    if token.isdigit():
        return [token]
    return []


def _parse_citation_group(group_text: str) -> List[str]:
    ids: List[str] = []
    if not group_text:
        return ids
    group_text = group_text.replace("\uFF1B", ",").replace(";", ",")
    for token in group_text.split(","):
        token = token.strip()
        if not token:
            continue
        ids.extend(_expand_numeric_tokens(token))
    return ids


def extract_numeric_citations(text: str) -> List[str]:
    """Extract numeric citation ids like [1], [1-3], (2,3)."""
    if not text:
        return []
    ids: List[str] = []

    # Bracketed citations
    for match in re.finditer(r"\[(.*?)\]", text):
        ids.extend(_parse_citation_group(match.group(1)))

    # Parentheses citations that look numeric-only
    for match in re.finditer(r"\(([^)]*?)\)", text):
        group = match.group(1)
        if re.search(r"[A-Za-z]", group):
            continue
        ids.extend(_parse_citation_group(group))

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out
