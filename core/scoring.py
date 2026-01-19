"""Quality scoring utilities (rule + LLM)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


CRITICAL_FIELDS = [
    'research_narrative.background.state_of_the_art_summary.value',
    'research_narrative.problem_formulation.research_gap.value',
    'research_narrative.problem_formulation.research_objectives.value',
    'research_narrative.methodology.approach_category.value',
    'research_narrative.methodology.method_description.value',
    'research_narrative.results_and_findings.key_findings.value',
]

SEGMENT_REQUIREMENTS = [
    (
        'research_narrative.problem_formulation.research_objectives.value',
        'research_narrative.problem_formulation.research_objectives.source_excerpt.segment_map',
    ),
    (
        'research_narrative.methodology.method_assumptions_and_limitations.assumptions.value',
        'research_narrative.methodology.method_assumptions_and_limitations.assumptions.source_excerpt.segment_map',
    ),
    (
        'research_narrative.methodology.method_assumptions_and_limitations.limitations.value',
        'research_narrative.methodology.method_assumptions_and_limitations.limitations.source_excerpt.segment_map',
    ),
    (
        'research_narrative.results_and_findings.key_findings.value',
        'research_narrative.results_and_findings.key_findings.source_excerpt.segment_map',
    ),
    (
        'research_narrative.discussion_and_conclusion.limitations_of_this_work.value',
        'research_narrative.discussion_and_conclusion.limitations_of_this_work.source_excerpt.segment_map',
    ),
    (
        'research_narrative.discussion_and_conclusion.future_work_suggestions.value',
        'research_narrative.discussion_and_conclusion.future_work_suggestions.source_excerpt.segment_map',
    ),
]


def _get_path(data: Dict[str, Any], path: str):
    cur = data
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _sentence_count(text: str) -> int:
    if not text:
        return 0
    parts = re.split(r'[???!?\n]+', text)
    return len([p for p in parts if p.strip()])


def rule_score(data: Dict[str, Any]) -> Tuple[int, List[str]]:
    issues: List[str] = []
    score = 100

    # Missing critical fields
    missing = []
    for path in CRITICAL_FIELDS:
        value = _get_path(data, path)
        if value in (None, '', {}, []):
            missing.append(path)
    if missing:
        score -= min(40, 8 * len(missing))
        issues.append(f"??????: {', '.join(missing)}")

    # Evidence coverage for segment_map
    evidence_penalties = 0
    for value_path, seg_path in SEGMENT_REQUIREMENTS:
        value = _get_path(data, value_path)
        if value in (None, '', {}, []):
            continue
        segs = _get_path(data, seg_path)
        if not isinstance(segs, list) or len(segs) == 0:
            evidence_penalties += 1

    if evidence_penalties:
        score -= min(30, 5 * evidence_penalties)
        issues.append("????????segment_map??")

    # Readability: ensure multi-sentence outputs where required
    key_findings = _get_path(data, 'research_narrative.results_and_findings.key_findings.value')
    if isinstance(key_findings, str) and _sentence_count(key_findings) < 1:
        score -= 10
        issues.append("????????????")

    score = max(0, min(100, score))
    return score, issues
