"""Evidence segmenter with paragraph-level granularity."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from core.citations import extract_numeric_citations


_HEADING_RE = re.compile(
    r"^(#+\s+.+|\d+(?:\.\d+)*\s+.+|[IVXLC]+\.?\s+.+|\\section\{.+\})$",
    re.IGNORECASE,
)


def _iter_paragraph_blocks(text: str) -> List[Tuple[int, int, str]]:
    blocks: List[Tuple[int, int, str]] = []
    if not text:
        return blocks
    lines = text.splitlines(keepends=True)
    offset = 0
    start = None
    buffer: List[str] = []
    for line in lines:
        if line.strip() == "":
            if start is not None:
                block_text = "".join(buffer)
                blocks.append((start, offset, block_text))
                buffer = []
                start = None
        else:
            if start is None:
                start = offset
            buffer.append(line)
        offset += len(line)
    if start is not None:
        blocks.append((start, offset, "".join(buffer)))
    return blocks


def _split_long_block(text: str, start_offset: int, max_chars: int) -> List[Tuple[int, int, str]]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [(start_offset, start_offset + len(text), text)]
    parts: List[Tuple[int, int, str]] = []
    chunk_start = 0
    last_break = 0
    for idx, ch in enumerate(text):
        if ch in {".", "!", "?", "。", "！", "？", "\n"}:
            last_break = idx + 1
        if idx - chunk_start + 1 >= max_chars:
            cut = last_break if last_break > chunk_start else idx + 1
            parts.append((start_offset + chunk_start, start_offset + cut, text[chunk_start:cut]))
            chunk_start = cut
            last_break = chunk_start
    if chunk_start < len(text):
        parts.append((start_offset + chunk_start, start_offset + len(text), text[chunk_start:]))
    return parts


def _is_heading(text: str) -> bool:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return False
    if len(cleaned) > 120:
        return False
    if _HEADING_RE.match(cleaned):
        return True
    if not re.search(r"[。！？.!?]$", cleaned) and len(cleaned.split()) <= 10:
        return True
    return False


def build_evidence_segments(
    text: str,
    min_chars: int = 200,
    max_chars: int = 1200,
) -> List[Dict[str, object]]:
    blocks = _iter_paragraph_blocks(text)
    chunks: List[Tuple[int, int, str]] = []
    pending_heading: Tuple[int, int, str] | None = None

    for start, end, block_text in blocks:
        cleaned = block_text.strip()
        if not cleaned:
            continue
        if _is_heading(cleaned):
            if pending_heading:
                ph_start, _ph_end, ph_text = pending_heading
                pending_heading = (ph_start, end, ph_text + "\n" + cleaned)
            else:
                pending_heading = (start, end, cleaned)
            continue
        if pending_heading:
            ph_start, _ph_end, ph_text = pending_heading
            cleaned = ph_text + "\n" + cleaned
            start = ph_start
            pending_heading = None

        for s_start, s_end, s_text in _split_long_block(cleaned, start, max_chars):
            chunks.append((s_start, s_end, s_text.strip()))

    if pending_heading:
        ph_start, ph_end, ph_text = pending_heading
        chunks.append((ph_start, ph_end, ph_text.strip()))

    merged: List[Tuple[int, int, str]] = []
    buffer: Tuple[int, int, str] | None = None
    for start, end, chunk_text in chunks:
        if not chunk_text:
            continue
        if buffer is None:
            buffer = (start, end, chunk_text)
            continue
        buf_start, buf_end, buf_text = buffer
        if len(buf_text) < min_chars:
            buffer = (buf_start, end, buf_text + "\n" + chunk_text)
        else:
            merged.append(buffer)
            buffer = (start, end, chunk_text)
    if buffer is not None:
        merged.append(buffer)

    if len(merged) >= 2 and len(merged[-1][2]) < min_chars:
        last_start, last_end, last_text = merged[-1]
        prev_start, prev_end, prev_text = merged[-2]
        merged[-2] = (prev_start, last_end, prev_text + "\n" + last_text)
        merged.pop()

    segments: List[Dict[str, object]] = []
    seg_index = 0
    for start, end, seg_text in merged:
        if not seg_text.strip():
            continue
        seg_index += 1
        citations = extract_numeric_citations(seg_text)
        segments.append(
            {
                "id": f"S{seg_index:04d}",
                "index": seg_index - 1,
                "start": start,
                "end": end,
                "text": seg_text,
                "section": "unknown",
                "citations": citations,
            }
        )
    return segments
