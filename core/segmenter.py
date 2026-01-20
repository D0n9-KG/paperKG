"""Evidence segmenter that preserves full-text coverage."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from core.citations import extract_numeric_citations


SECTION_LABELS = (
    "background",
    "problem_formulation",
    "methodology",
    "results_and_findings",
    "discussion_and_conclusion",
    "references",
    "unknown",
)


_SECTION_KEYWORDS: Dict[str, List[str]] = {
    "background": [
        "introduction",
        "background",
        "related work",
        "literature",
        "overview",
        "motivation",
        "\u5f15\u8a00",
        "\u80cc\u666f",
        "\u76f8\u5173\u5de5\u4f5c",
        "\u6587\u732e\u7efc\u8ff0",
    ],
    "problem_formulation": [
        "problem",
        "objective",
        "aim",
        "goal",
        "question",
        "\u7814\u7a76\u95ee\u9898",
        "\u7814\u7a76\u76ee\u6807",
        "\u7814\u7a76\u7a7a\u767d",
        "\u76ee\u7684",
    ],
    "methodology": [
        "method",
        "methods",
        "methodology",
        "materials",
        "experiment",
        "simulation",
        "model",
        "approach",
        "\u65b9\u6cd5",
        "\u6750\u6599",
        "\u5b9e\u9a8c",
        "\u4eff\u771f",
        "\u6a21\u578b",
    ],
    "results_and_findings": [
        "result",
        "results",
        "finding",
        "analysis",
        "observations",
        "\u7ed3\u679c",
        "\u53d1\u73b0",
        "\u5206\u6790",
    ],
    "discussion_and_conclusion": [
        "discussion",
        "conclusion",
        "limitations",
        "future work",
        "summary",
        "\u8ba8\u8bba",
        "\u7ed3\u8bba",
        "\u5c40\u9650",
        "\u672a\u6765\u5de5\u4f5c",
        "\u603b\u7ed3",
    ],
    "references": [
        "references",
        "bibliography",
        "\u53c2\u8003\u6587\u732e",
    ],
}

SENTENCE_PUNCT = re.compile(r"[\u3002\uff01\uff1f.!?]")


def _normalize_heading(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _guess_section_from_heading(heading: str) -> str:
    normalized = _normalize_heading(heading)
    for section, keywords in _SECTION_KEYWORDS.items():
        for kw in keywords:
            if kw in normalized:
                return section
    return "unknown"


def _detect_heading_positions(text: str) -> List[Tuple[int, str, str]]:
    positions: List[Tuple[int, str, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            md_match = re.match(r"^\\s*#+\\s*(.+)$", stripped)
            if md_match:
                heading = md_match.group(1)
                positions.append((offset, _guess_section_from_heading(heading), heading))
            else:
                # Numbered headings like "1. Introduction"
                num_match = re.match(r"^\\s*\\d+\\.?\\s+(.+)$", stripped)
                if num_match and len(stripped) <= 120:
                    heading = num_match.group(1)
                    positions.append((offset, _guess_section_from_heading(heading), heading))
        offset += len(line)
    return positions


def _section_for_offset(offset: int, headings: List[Tuple[int, str, str]]) -> str:
    current = "unknown"
    for pos, section, _heading in headings:
        if pos <= offset:
            current = section or "unknown"
        else:
            break
    return current


def _iter_sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    if not text:
        return spans
    pattern = re.compile(r".*?(?:[。！？.!?]+|\n\n+|$)", re.S)
    for match in pattern.finditer(text):
        start, end = match.span()
        if start == end:
            continue
        spans.append((start, end))
    # Ensure full coverage
    if spans:
        last_end = spans[-1][1]
        if last_end < len(text):
            spans.append((last_end, len(text)))
    else:
        spans.append((0, len(text)))
    return spans


def _should_merge_segment(segment_text: str) -> bool:
    cleaned = segment_text.strip()
    if not cleaned:
        return True
    if not SENTENCE_PUNCT.search(cleaned):
        return True
    return False


def _merge_spans(text: str, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return spans
    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = spans[0]
    for next_start, next_end in spans[1:]:
        current_text = text[cur_start:cur_end]
        if _should_merge_segment(current_text):
            cur_end = next_end
            continue
        merged.append((cur_start, cur_end))
        cur_start, cur_end = next_start, next_end
    merged.append((cur_start, cur_end))
    return merged


def build_evidence_segments(text: str) -> List[Dict[str, object]]:
    headings = _detect_heading_positions(text)
    spans = _iter_sentence_spans(text)
    spans = _merge_spans(text, spans)
    segments: List[Dict[str, object]] = []
    seg_index = 0
    for start, end in spans:
        seg_text = text[start:end]
        if not seg_text.strip():
            continue
        seg_index += 1
        section = _section_for_offset(start, headings)
        citations = extract_numeric_citations(seg_text)
        segments.append(
            {
                "id": f"S{seg_index:04d}",
                "index": seg_index - 1,
                "start": start,
                "end": end,
                "text": seg_text,
                "section": section,
                "citations": citations,
            }
        )
    return segments
