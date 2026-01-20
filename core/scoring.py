"""Quality scoring utilities (rule + LLM)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


CRITICAL_LIST_SECTIONS = [
    "background",
    "research_gaps",
]

CRITICAL_MAIN_SECTIONS = [
    "methods",
    "results",
    "conclusions",
]


def _iter_logic_items(narrative: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for section in ("background", "research_gaps", "research_questions", "hypotheses"):
        entries = narrative.get(section)
        if isinstance(entries, list):
            items.extend([x for x in entries if isinstance(x, dict)])
    for block_name in ("methods", "results", "conclusions"):
        block = narrative.get(block_name)
        if not isinstance(block, dict):
            continue
        main = block.get("main")
        if isinstance(main, dict):
            items.append(main)
        supports = block.get("supports")
        if isinstance(supports, list):
            items.extend([x for x in supports if isinstance(x, dict)])
    return items


def rule_score(data: Dict[str, Any]) -> Tuple[int, List[str]]:
    issues: List[str] = []
    score = 100

    narrative = data.get("research_narrative") if isinstance(data, dict) else None
    if not isinstance(narrative, dict):
        return 0, ["research_narrative missing"]

    # Missing critical list sections
    missing = []
    for section in CRITICAL_LIST_SECTIONS:
        items = narrative.get(section)
        if not isinstance(items, list) or len(items) == 0:
            missing.append(section)
    if missing:
        score -= min(40, 8 * len(missing))
        issues.append(f"missing critical lists: {', '.join(missing)}")

    # Missing critical main nodes
    missing_main = []
    for block_name in CRITICAL_MAIN_SECTIONS:
        block = narrative.get(block_name)
        main = block.get("main") if isinstance(block, dict) else None
        if not isinstance(main, dict) or main.get("value") in (None, "", [], {}):
            missing_main.append(block_name)
    if missing_main:
        score -= min(30, 8 * len(missing_main))
        issues.append(f"missing main nodes: {', '.join(missing_main)}")

    # Evidence coverage
    evidence_penalties = 0
    for item in _iter_logic_items(narrative):
        evidence_ids = item.get("evidence_segment_ids")
        if not isinstance(evidence_ids, list) or len(evidence_ids) == 0:
            evidence_penalties += 1
    if evidence_penalties:
        score -= min(30, 3 * evidence_penalties)
        issues.append("some statements missing evidence_segment_ids")

    # Logic chains
    chains = narrative.get("logic_chains")
    if not isinstance(chains, list) or len(chains) == 0:
        score -= 15
        issues.append("logic_chains missing or empty")

    score = max(0, min(100, score))
    return score, issues

