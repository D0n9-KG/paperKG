"""Main extraction pipeline."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.config import Config
from core.schema import SchemaLoader
from core.validation import SchemaValidator
from core.scoring import rule_score
from core.crossref import CrossrefClient, CrossrefDataExtractor
from core.keywords import extract_keywords
from core.segmenter import build_evidence_segments
from llm.prompts import (
    METADATA_PROMPT_BASE,
    MULTIMEDIA_CONTENT_PROMPT_BASE,
    JSON_SCHEMA_REPAIR_PROMPT,
    QUALITY_RATER_PROMPT,
    CONTENT_REWRITE_PROMPT,
    KEYWORDS_PROMPT,
    CITATION_PURPOSE_PROMPT,
    EVIDENCE_SCAN_PROMPT,
    MAIN_ID_PICK_PROMPT,
    RESEARCH_NARRATIVE_SYNTH_PROMPT,
    STATE_GAP_OBJECTIVE_LINK_PROMPT,
)
from llm.prompt_builder import PromptBuilder
from llm.providers.factory import build_client

try:
    import json_repair
    JSON_REPAIR_AVAILABLE = True
except Exception:
    JSON_REPAIR_AVAILABLE = False
    json_repair = None

logger = logging.getLogger(__name__)


class PaperKGExtractor:
    def __init__(self, config_path: str = "config/default.yaml", schema_path: str = "config/output_schema.json"):
        self.config = Config(config_path)
        self.schema_loader = SchemaLoader(schema_path)
        self.prompt_builder = PromptBuilder(self.schema_loader)
        self.validator = SchemaValidator(self.schema_loader.schema)

        self.workflow = self.config.get('workflow', {})
        self.context = self.config.get('context', {})
        self.output_cfg = self.config.get('output', {})
        self.crossref_cfg = self.config.get('crossref', {})
        self.quality_cfg = self.config.get('quality', {})
        self.neo4j_cfg = self.config.get('neo4j', {})
        self.keywords_cfg = self.config.get('keywords', {})
        self.citation_purpose_cfg = self.config.get('citation_purpose', {})
        self.metadata_cfg = self.config.get('metadata', {})
        self.multimedia_cfg = self.config.get('multimedia', {})
        self.iterative_cfg = self.workflow.get('iterative_refine', {})
        self._neo4j_cleared = False

        self.crossref_client = CrossrefClient(self.crossref_cfg) if self.crossref_cfg.get('enable', True) else None

    async def _call_agent(self, agent_name: str, base_prompt: str, text: str, strict_json: bool = False) -> Dict[str, Any]:
        agent_cfg = self.config.agent_configs.get(agent_name, {})
        provider = agent_cfg.get('provider', self.config.default_provider)
        model = agent_cfg.get('model')
        max_tokens = agent_cfg.get('max_tokens', 4000)
        temperature = agent_cfg.get('temperature', 0.3)

        client = build_client(self.config, provider)
        prompt = base_prompt.replace("{text}", text)
        prompt = self.prompt_builder.build(agent_name, prompt)
        prompt += (
            "\n\n【质量与可读性要求】\n"
            "- value必须由完整可读的句子组成（包含主语/动作/对象或条件）\n"
            "- 合并字段必须多句且逐句提供segment_map证据（无证据句子必须删除）\n"
            "- 合并字段的source_excerpt必须是对象，包含text与segment_map，严禁输出为字符串\n"
            "- 不要臆造，不要使用空泛表述\n"
            "- 允许更详细，但必须忠实原文\n"
        )

        if strict_json:
            prompt += (
                "\n\n【严格JSON输出】\n"
                "- 只输出JSON，不要解释、标题、注释或Markdown。\n"
                "- 不要输出任何非JSON字符或多余文本。\n"
            )

        response_format = None
        extra_body = None
        structured_cfg = agent_cfg.get('structured_output', {})
        if structured_cfg.get('enable'):
            schema = self.prompt_builder.get_schema(agent_name)
            if schema:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": structured_cfg.get('schema_name', agent_name),
                        "schema": schema,
                        "strict": bool(structured_cfg.get('strict', True)),
                    }
                }
                if provider == "openrouter":
                    extra_body = {"require_parameters": True}

        response = await client.call(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=self.workflow.get('retry_attempts', 3),
            retry_delay=self.workflow.get('retry_delay', 2),
            response_format=response_format,
            extra_body=extra_body,
        )
        parsed = self._parse_json(response)
        return parsed or {}

    def _parse_json(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            if JSON_REPAIR_AVAILABLE:
                try:
                    repaired = json_repair.repair_json(response)
                    return json.loads(repaired)
                except Exception:
                    return {}
            return {}

    async def _repair_json_with_llm(self, data: Dict[str, Any], paper_text: str) -> Dict[str, Any]:
        agent_cfg = self.config.agent_configs.get('json_repair', {})
        provider = agent_cfg.get('provider', self.config.default_provider)
        model = agent_cfg.get('model')
        max_tokens = agent_cfg.get('max_tokens', 4000)
        temperature = agent_cfg.get('temperature', 0.1)

        client = build_client(self.config, provider)

        errors = self.validator.validate(data)
        if not errors:
            return data

        max_repair_chars = int(self.config.get('repair', {}).get('max_repair_chars', 40000))
        data_text = json.dumps(data, ensure_ascii=False, indent=2)
        if len(data_text) > max_repair_chars:
            logger.warning("Skip JSON repair: payload too large")
            return data

        prompt = JSON_SCHEMA_REPAIR_PROMPT.replace("{validation_errors}", "\n".join(errors))
        prompt = prompt.replace("{json_data}", data_text)

        response = await client.call(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=self.workflow.get('retry_attempts', 3),
            retry_delay=self.workflow.get('retry_delay', 2),
        )
        repaired = self._parse_json(response)
        if not repaired:
            return data
        return self._merge_preserve(data, repaired)

    async def _score_with_llm(self, data: Dict[str, Any], paper_text: str) -> Dict[str, Any]:
        agent_cfg = self.config.agent_configs.get('quality_rater', {})
        provider = agent_cfg.get('provider', self.config.default_provider)
        model = agent_cfg.get('model')
        max_tokens = agent_cfg.get('max_tokens', 2000)
        temperature = agent_cfg.get('temperature', 0.0)
        client = build_client(self.config, provider)

        prompt = QUALITY_RATER_PROMPT.replace("{paper_text}", paper_text)
        prompt = prompt.replace("{extracted_json}", json.dumps(data, ensure_ascii=False, indent=2))

        response = await client.call(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=self.workflow.get('retry_attempts', 3),
            retry_delay=self.workflow.get('retry_delay', 2),
        )
        return self._parse_json(response) or {"score": 0, "issues": ["invalid_score"], "needs_refine": True}

    async def _compute_quality_score(self, data: Dict[str, Any], paper_text: str) -> int:
        rule_sc, _ = rule_score(data)
        if not self.quality_cfg.get('use_llm', True):
            return rule_sc
        llm_sc = None
        llm_res = await self._score_with_llm(data, paper_text)
        if isinstance(llm_res, dict):
            llm_sc = llm_res.get('score')
        weight_r = float(self.quality_cfg.get('rule_weight', 0.4))
        weight_l = float(self.quality_cfg.get('llm_weight', 0.6))
        return rule_sc if llm_sc is None else int(rule_sc * weight_r + llm_sc * weight_l)

    def _front_text(self, text: str) -> str:
        ratio = float(self.metadata_cfg.get('llm_window_ratio', 0.15))
        max_chars = int(self.metadata_cfg.get('llm_max_chars', 60000))
        if ratio <= 0:
            ratio = 0.15
        limit = min(len(text), max_chars, max(1000, int(len(text) * ratio)))
        return text[:limit]

    @staticmethod
    def _prune_metadata_unrelated(meta: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(meta, dict):
            return meta
        meta = dict(meta)
        # Remove Crossref/process-only fields
        meta.pop("metadata_source", None)
        meta.pop("citation_metrics", None)
        dates = meta.get("dates")
        if isinstance(dates, dict):
            dates = dict(dates)
            for key in ("created", "deposited", "indexed"):
                dates.pop(key, None)
            meta["dates"] = dates
        return meta

    def _build_evidence_segments(self, text: str) -> List[Dict[str, Any]]:
        seg_cfg = self.config.get("segmenter", {}) or {}
        if not isinstance(seg_cfg, dict):
            seg_cfg = {}
        min_chars = int(seg_cfg.get("min_chars", 200))
        max_chars = int(seg_cfg.get("max_chars", 1200))
        return build_evidence_segments(text, min_chars=min_chars, max_chars=max_chars)

    @staticmethod
    def _clean_segment_text(text: str) -> str:
        return " ".join(text.split()) if isinstance(text, str) else ""

    @staticmethod
    def _build_citation_entries(citations: List[Any]) -> List[Dict[str, Any]]:
        if not isinstance(citations, list):
            return []
        entries = []
        for cite in citations:
            if cite is None:
                continue
            cid = str(cite).strip()
            if not cid:
                continue
            entries.append({"citation_id": cid, "citation_text": None, "purpose": ""})
        return entries

    def _backfill_narrative_with_selected_ids(
        self,
        narrative: Dict[str, Any],
        selected_ids: Dict[str, List[str]],
        segments_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(narrative, dict):
            return narrative
        if not isinstance(selected_ids, dict):
            return narrative

        id_to_item: Dict[str, Dict[str, Any]] = {}
        for sid, item in segments_by_id.items():
            if isinstance(sid, str) and isinstance(item, dict):
                id_to_item[sid] = item

        def _existing_ids(items: List[Dict[str, Any]]) -> set:
            ids = set()
            for item in items:
                if not isinstance(item, dict):
                    continue
                evid = item.get("evidence_segment_ids")
                if isinstance(evid, list):
                    for sid in evid:
                        if isinstance(sid, str):
                            ids.add(sid)
            return ids

        def _make_item(seg_id: str) -> Optional[Dict[str, Any]]:
            item = id_to_item.get(seg_id)
            if not isinstance(item, dict):
                return None
            text = self._clean_segment_text(item.get("text", ""))
            if not text:
                return None
            citations = self._build_citation_entries(item.get("citations", []))
            return {
                "value": text.strip(),
                "evidence_segment_ids": [seg_id],
                "citations": citations,
            }

        # List sections backfill
        list_mapping = {
            "background": "background_ids",
            "state_of_art": "state_of_art_ids",
            "research_gaps": "research_gap_ids",
            "research_questions": "research_question_ids",
            "research_objectives": "objective_ids",
            "hypotheses": "hypothesis_ids",
        }
        for section, sel_key in list_mapping.items():
            items = narrative.get(section)
            if not isinstance(items, list):
                items = []
            existing = _existing_ids(items)
            selected = selected_ids.get(sel_key, [])
            if not isinstance(selected, list):
                selected = []
            for sid in selected:
                if not isinstance(sid, str) or sid in existing:
                    continue
                new_item = _make_item(sid)
                if new_item:
                    items.append(new_item)
                    existing.add(sid)
            narrative[section] = items

        # Main/support blocks backfill
        block_mapping = (
            ("methods", "method_main_ids", "method_support_ids"),
            ("results", "result_main_ids", "result_support_ids"),
            ("conclusions", "conclusion_main_ids", "conclusion_support_ids"),
        )
        for block_name, main_key, support_key in block_mapping:
            block = narrative.get(block_name)
            if not isinstance(block, dict):
                block = {"main": None, "supports": []}
            main = block.get("main")
            supports = block.get("supports")
            if not isinstance(supports, list):
                supports = []
            existing_support_ids = _existing_ids(supports)
            if isinstance(main, dict):
                existing_support_ids |= _existing_ids([main])

            main_ids = selected_ids.get(main_key, [])
            if (main is None or not isinstance(main, dict)) and isinstance(main_ids, list) and main_ids:
                main_item = _make_item(main_ids[0])
                if main_item:
                    block["main"] = main_item
                    existing_support_ids.add(main_ids[0])

            support_ids = selected_ids.get(support_key, [])
            if isinstance(support_ids, list):
                for sid in support_ids:
                    if not isinstance(sid, str) or sid in existing_support_ids:
                        continue
                    new_item = _make_item(sid)
                    if new_item:
                        supports.append(new_item)
                        existing_support_ids.add(sid)

            block["supports"] = supports
            narrative[block_name] = block

        return narrative

    def _selection_keys(self) -> List[str]:
        return [
            "background_ids",
            "state_of_art_ids",
            "research_gap_ids",
            "research_question_ids",
            "objective_ids",
            "hypothesis_ids",
            "method_main_ids",
            "method_support_ids",
            "result_main_ids",
            "result_support_ids",
            "conclusion_main_ids",
            "conclusion_support_ids",
        ]

    def _normalize_selection_payload(self, payload: Any) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = {key: [] for key in self._selection_keys()}
        if not isinstance(payload, dict):
            return normalized
        for key in self._selection_keys():
            items = payload.get(key, [])
            if isinstance(items, list):
                normalized[key] = [str(i).strip() for i in items if isinstance(i, (str, int)) and str(i).strip()]
        return normalized

    def _chunk_segment_payloads(self, segments: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        narrative_cfg = self.config.get("narrative", {}) if isinstance(self.config.get("narrative", {}), dict) else {}
        scan_cfg = narrative_cfg.get("evidence_scan", {}) if isinstance(narrative_cfg.get("evidence_scan", {}), dict) else {}
        max_chars = int(scan_cfg.get("max_chars", self.context.get("evidence_max_chars", 20000)))
        max_segments = int(scan_cfg.get("max_segments_per_call", 40))
        text_max = int(scan_cfg.get("text_max_chars", 800))

        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_chars = 0
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            seg_id = seg.get("id")
            text = self._clean_segment_text(seg.get("text", ""))
            if not isinstance(seg_id, str) or not text:
                continue
            if len(text) > text_max:
                text = text[:text_max]
            item = {"id": seg_id, "text": text}
            item_len = len(text) + 20
            if current and (len(current) >= max_segments or current_chars + item_len > max_chars):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += item_len
        if current:
            batches.append(current)
        return batches

    async def _select_evidence_ids(
        self,
        segments: List[Dict[str, Any]],
        agent_name: str = "research_narrative_selector",
    ) -> Dict[str, List[str]]:
        batches = self._chunk_segment_payloads(segments)
        if not batches:
            return {key: [] for key in self._selection_keys()}

        api_limit = int(self.workflow.get("api_concurrent_limit", 6))
        semaphore = asyncio.Semaphore(max(1, api_limit))

        async def _run_batch(batch: List[Dict[str, Any]]) -> Dict[str, List[str]]:
            payload = json.dumps({"segments": batch}, ensure_ascii=False, indent=2)
            try:
                parsed = await self._call_agent(agent_name, EVIDENCE_SCAN_PROMPT, payload, strict_json=True)
            except Exception as exc:
                logger.warning(f"Evidence scan failed: {exc}")
                return {key: [] for key in self._selection_keys()}
            return self._normalize_selection_payload(parsed)

        async def _guarded(batch: List[Dict[str, Any]]) -> Dict[str, List[str]]:
            async with semaphore:
                return await _run_batch(batch)

        results = await asyncio.gather(*[_guarded(batch) for batch in batches])
        merged: Dict[str, List[str]] = {key: [] for key in self._selection_keys()}
        seen: Dict[str, set] = {key: set() for key in self._selection_keys()}
        for result in results:
            for key in self._selection_keys():
                for sid in result.get(key, []):
                    if sid not in seen[key]:
                        seen[key].add(sid)
                        merged[key].append(sid)
        return merged

    async def _resolve_main_ids(
        self,
        selections: Dict[str, List[str]],
        segments_by_id: Dict[str, Dict[str, Any]],
        agent_name: str = "research_narrative_selector",
    ) -> Dict[str, List[str]]:
        narrative_cfg = self.config.get("narrative", {}) if isinstance(self.config.get("narrative", {}), dict) else {}
        picker_cfg = narrative_cfg.get("main_picker", {}) if isinstance(narrative_cfg.get("main_picker", {}), dict) else {}
        max_candidates = int(picker_cfg.get("max_candidates", 40))
        text_max = int(picker_cfg.get("text_max_chars", 800))

        def _build_candidates(main_key: str, support_key: str) -> List[Dict[str, str]]:
            candidates = selections.get(main_key) or []
            if not candidates:
                candidates = selections.get(support_key) or []
            trimmed: List[Dict[str, str]] = []
            for sid in candidates:
                seg = segments_by_id.get(sid)
                if not seg:
                    continue
                text = self._clean_segment_text(seg.get("text", ""))
                if not text:
                    continue
                if len(text) > text_max:
                    text = text[:text_max]
                trimmed.append({"id": sid, "text": text})
                if len(trimmed) >= max_candidates:
                    break
            return trimmed

        method_candidates = _build_candidates("method_main_ids", "method_support_ids")
        result_candidates = _build_candidates("result_main_ids", "result_support_ids")
        conclusion_candidates = _build_candidates("conclusion_main_ids", "conclusion_support_ids")

        if not (method_candidates or result_candidates or conclusion_candidates):
            return selections

        payload = json.dumps(
            {
                "method_candidates": method_candidates,
                "result_candidates": result_candidates,
                "conclusion_candidates": conclusion_candidates,
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            parsed = await self._call_agent(agent_name, MAIN_ID_PICK_PROMPT, payload, strict_json=True)
        except Exception as exc:
            logger.warning(f"Main id pick failed: {exc}")
            parsed = {}

        def _pick(field: str, candidates: List[Dict[str, str]]) -> Optional[str]:
            if isinstance(parsed, dict):
                picked = parsed.get(field)
                if isinstance(picked, str) and picked in segments_by_id:
                    return picked
            if candidates:
                return candidates[0].get("id")
            return None

        method_main = _pick("method_main_id", method_candidates)
        result_main = _pick("result_main_id", result_candidates)
        conclusion_main = _pick("conclusion_main_id", conclusion_candidates)

        def _apply(main_key: str, support_key: str, main_id: Optional[str]) -> None:
            if main_id:
                selections[main_key] = [main_id]
                supports = selections.get(support_key, [])
                if isinstance(supports, list) and main_id in supports:
                    selections[support_key] = [sid for sid in supports if sid != main_id]

        _apply("method_main_ids", "method_support_ids", method_main)
        _apply("result_main_ids", "result_support_ids", result_main)
        _apply("conclusion_main_ids", "conclusion_support_ids", conclusion_main)
        return selections

    def _extract_doi_from_text(self, text: str) -> Optional[str]:
        import re
        match = re.search(r"(10\\.[0-9]{4,9}/[^\\s\"<>]+)", text, flags=re.IGNORECASE)
        if not match:
            return None
        doi = match.group(1).rstrip(").,;")
        return doi.strip()

    def _metadata_missing_fields(self, metadata: Dict[str, Any]) -> List[str]:
        required = self.metadata_cfg.get('required_fields', [])
        missing = []
        for field in required:
            value = metadata.get(field)
            if value in (None, "", [], {}):
                missing.append(field)
        return missing

    async def _extract_metadata(self, paper_text: str) -> Dict[str, Any]:
        front_text = self._front_text(paper_text)
        metadata: Dict[str, Any] = {}

        if self.crossref_cfg.get('enable', True) and self.metadata_cfg.get('use_crossref_first', True):
            doi = self._extract_doi_from_text(paper_text)
            crossref_data = None
            if doi:
                crossref_data = await self.crossref_client.get_metadata(doi=doi)
            if not crossref_data:
                try:
                    seed = await self._call_agent('metadata', METADATA_PROMPT_BASE, front_text, strict_json=True)
                except Exception as exc:
                    logger.warning(f"Metadata seed extraction failed: {exc}")
                    seed = {}
                title = seed.get('title') if isinstance(seed, dict) else None
                authors = []
                if isinstance(seed, dict):
                    for author in seed.get('authors', []) if isinstance(seed.get('authors'), list) else []:
                        if isinstance(author, dict):
                            name = author.get('full_name') or author.get('family_name') or author.get('given_name')
                            if name:
                                authors.append(str(name))
                year = seed.get('publication_year') if isinstance(seed, dict) else None
                crossref_data = await self.crossref_client.get_metadata(title=title, authors=authors, year=year)
            if crossref_data:
                metadata = CrossrefDataExtractor.extract_complete_metadata(crossref_data)

        try:
            llm_meta = await self._call_agent('metadata', METADATA_PROMPT_BASE, front_text, strict_json=True)
        except Exception as exc:
            logger.warning(f"Metadata LLM extraction failed: {exc}")
            llm_meta = {}

        if isinstance(llm_meta, dict):
            if metadata:
                metadata = self._merge_preserve(llm_meta, metadata)
                if isinstance(metadata.get("metadata_source"), dict):
                    metadata["metadata_source"]["llm_supplemented"] = True
            else:
                metadata = llm_meta

        if self.keywords_cfg.get('enable', True):
            if not metadata.get('keywords'):
                max_keywords = int(self.keywords_cfg.get('max_keywords', 12))
                language = self.keywords_cfg.get('language', 'en')
                extracted = extract_keywords(front_text, max_keywords=max_keywords, language=language)
                if not extracted and self.keywords_cfg.get('use_llm', True):
                    try:
                        extracted = await self._extract_keywords_with_llm(front_text, max_keywords)
                    except Exception as exc:
                        logger.warning(f"Keyword LLM fallback failed: {exc}")
                        extracted = []
                if extracted:
                    metadata['keywords'] = extracted

        missing = self._metadata_missing_fields(metadata)
        if missing:
            logger.warning(f"Metadata missing fields: {', '.join(missing)}")
            if self.metadata_cfg.get('retry_if_missing', True):
                try:
                    llm_retry = await self._call_agent('metadata', METADATA_PROMPT_BASE, front_text, strict_json=True)
                    if isinstance(llm_retry, dict):
                        metadata = self._merge_preserve(llm_retry, metadata)
                        if isinstance(metadata.get("metadata_source"), dict):
                            metadata["metadata_source"]["llm_supplemented"] = True
                except Exception as exc:
                    logger.warning(f"Metadata retry failed: {exc}")

        return metadata

    def _has_figure_mentions(self, text: str) -> bool:
        import re
        return bool(re.search(r"\\bFigure\\b|\\bFig\\.?\\b|图\\s*\\d", text, flags=re.IGNORECASE))

    def _has_equation_mentions(self, text: str) -> bool:
        import re
        return bool(re.search(r"\\bEquation\\b|\\bEq\\.?\\b|公式", text, flags=re.IGNORECASE))

    def _multimedia_incomplete(self, multimedia: Dict[str, Any], paper_text: str) -> bool:
        references = multimedia.get('references', {}) if isinstance(multimedia, dict) else {}
        ref_list = references.get('reference_list', []) if isinstance(references, dict) else []
        if not isinstance(ref_list, list) or len(ref_list) == 0:
            return True

        images = multimedia.get('images', {}) if isinstance(multimedia, dict) else {}
        image_count = 0
        if isinstance(images, dict):
            for items in images.values():
                if isinstance(items, list):
                    image_count += len(items)
        require_images = bool(self.multimedia_cfg.get('require_images_if_figure_mentions', True))
        if require_images and self._has_figure_mentions(paper_text) and image_count == 0:
            return True

        formulas = multimedia.get('formulas', {}) if isinstance(multimedia, dict) else {}
        formula_list = formulas.get('formula_list', []) if isinstance(formulas, dict) else []
        require_formulas = bool(self.multimedia_cfg.get('require_formulas_if_equation_mentions', True))
        if require_formulas and self._has_equation_mentions(paper_text) and (not isinstance(formula_list, list) or len(formula_list) == 0):
            return True

        return False

    async def _extract_multimedia(self, paper_text: str) -> Dict[str, Any]:
        try:
            multimedia = await self._call_agent('multimedia_content', MULTIMEDIA_CONTENT_PROMPT_BASE, paper_text, strict_json=True)
        except Exception as exc:
            logger.warning(f"Multimedia extraction failed: {exc}")
            multimedia = {}

        if self.multimedia_cfg.get('retry_if_empty', True) and self._multimedia_incomplete(multimedia, paper_text):
            logger.warning("Multimedia incomplete, retrying once.")
            try:
                multimedia = await self._call_agent('multimedia_content', MULTIMEDIA_CONTENT_PROMPT_BASE, paper_text, strict_json=True)
            except Exception as exc:
                logger.warning(f"Multimedia retry failed: {exc}")
        return multimedia

    async def _extract_research_narrative(
        self,
        paper_text: str,
        evidence_segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        segments_by_id = {
            seg.get("id"): seg for seg in evidence_segments if isinstance(seg, dict) and seg.get("id")
        }
        selected_ids = await self._select_evidence_ids(evidence_segments)
        selected_ids = await self._resolve_main_ids(selected_ids, segments_by_id)
        selected_text = json.dumps(selected_ids, ensure_ascii=False, indent=2)

        selected_id_set = set()
        for key in self._selection_keys():
            for sid in selected_ids.get(key, []):
                if isinstance(sid, str):
                    selected_id_set.add(sid)
        selected_segments = []
        for seg in evidence_segments:
            sid = seg.get("id") if isinstance(seg, dict) else None
            if not isinstance(sid, str) or sid not in selected_id_set:
                continue
            selected_segments.append(
                {
                    "id": sid,
                    "text": self._clean_segment_text(seg.get("text", "")),
                    "citations": seg.get("citations", []),
                }
            )

        prompt = RESEARCH_NARRATIVE_SYNTH_PROMPT.replace(
            "{segments}", json.dumps(selected_segments, ensure_ascii=False, indent=2)
        )
        prompt = prompt.replace("{selected_ids}", selected_text)

        primary_ok = True
        try:
            narrative = await self._call_agent("research_narrative", prompt, "", strict_json=True)
        except Exception as exc:
            logger.warning(f"Research narrative extraction failed: {exc}")
            primary_ok = False
            try:
                narrative = await self._call_agent("research_narrative_fallback", prompt, "", strict_json=True)
                logger.warning("Research narrative fallback used: deepseek")
            except Exception as fallback_exc:
                logger.warning(f"Research narrative fallback failed: {fallback_exc}")
                narrative = {}

        if isinstance(narrative, dict):
            narrative = self._ensure_research_narrative_sections(narrative)
            if self._needs_narrative_retry(narrative):
                try:
                    retry_agent = "research_narrative" if primary_ok else "research_narrative_fallback"
                    retry_narrative = await self._call_agent(retry_agent, prompt, "", strict_json=True)
                    if isinstance(retry_narrative, dict):
                        narrative = self._merge_preserve(narrative, retry_narrative)
                except Exception as exc:
                    logger.warning(f"Research narrative retry failed: {exc}")
            narrative = self._ensure_research_narrative_sections(narrative)
            narrative = self._backfill_narrative_with_selected_ids(narrative, selected_ids, segments_by_id)
            return self._ensure_research_narrative_sections(narrative)
        return {}

    def _needs_narrative_retry(self, narrative: Dict[str, Any]) -> bool:
        narrative_cfg = self.config.get("narrative", {})
        min_cfg = narrative_cfg.get("min_counts", {}) if isinstance(narrative_cfg, dict) else {}
        default_min = {"background": 1, "research_gaps": 1, "research_questions": 0, "research_objectives": 1, "hypotheses": 0}
        for key, default_val in default_min.items():
            target_min = int(min_cfg.get(key, default_val))
            if target_min <= 0:
                continue
            items = narrative.get(key)
            if not isinstance(items, list) or len(items) < target_min:
                return True

        methods = narrative.get("methods", {})
        results = narrative.get("results", {})
        conclusions = narrative.get("conclusions", {})
        main_nodes = [
            methods.get("main") if isinstance(methods, dict) else None,
            results.get("main") if isinstance(results, dict) else None,
            conclusions.get("main") if isinstance(conclusions, dict) else None,
        ]
        for node in main_nodes:
            if node is None:
                return True
            if not isinstance(node, dict):
                return True
            value = node.get("value")
            if value in (None, "", [], {}):
                return True
        chains = narrative.get("logic_chains")
        if not isinstance(chains, list) or len(chains) == 0:
            return True
        return False

    @staticmethod
    def _ensure_research_narrative_sections(narrative: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(narrative, dict):
            return {}
        defaults = {
            "background": [],
            "state_of_art": [],
            "research_gaps": [],
            "research_questions": [],
            "research_objectives": [],
            "hypotheses": [],
            "background_hub": None,
            "state_hub": None,
            "gap_hub": None,
            "objective_hub": None,
            "hypothesis_hub": None,
            "methods": {"main": None, "supports": []},
            "results": {"main": None, "supports": []},
            "conclusions": {"main": None, "supports": []},
            "logic_chains": [],
            "node_relations": [],
        }
        for key, default in defaults.items():
            if key not in narrative or not isinstance(narrative.get(key), type(default)):
                narrative[key] = default
        # Normalize nested structures
        for section in ("methods", "results", "conclusions"):
            block = narrative.get(section)
            if not isinstance(block, dict):
                narrative[section] = {"main": None, "supports": []}
                continue
            if "main" not in block:
                block["main"] = None
            if not isinstance(block.get("supports"), list):
                block["supports"] = []
        return narrative

    async def _refine_research_narrative(self, narrative: Dict[str, Any], paper_text: str) -> Dict[str, Any]:
        if not narrative:
            return narrative
        try:
            prompt = CONTENT_REWRITE_PROMPT.replace("{paper_text}", paper_text)
            prompt = prompt.replace("{extracted_json}", json.dumps({"research_narrative": narrative}, ensure_ascii=False, indent=2))
            agent_cfg = self.config.agent_configs.get('content_refiner', {})
            provider = agent_cfg.get('provider', self.config.default_provider)
            model = agent_cfg.get('model')
            max_tokens = agent_cfg.get('max_tokens', 6000)
            temperature = agent_cfg.get('temperature', 0.2)
            client = build_client(self.config, provider)
            response = await client.call(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                retries=self.workflow.get('retry_attempts', 3),
                retry_delay=self.workflow.get('retry_delay', 2),
            )
            parsed = self._parse_json(response)
            if isinstance(parsed, dict) and isinstance(parsed.get('research_narrative'), dict):
                return self._ensure_research_narrative_sections(parsed.get('research_narrative'))
        except Exception as exc:
            logger.warning(f"Research narrative refine failed: {exc}")
        return self._ensure_research_narrative_sections(narrative)

    async def _refine_with_llm(self, data: Dict[str, Any], paper_text: str) -> Dict[str, Any]:
        agent_cfg = self.config.agent_configs.get('content_refiner', {})
        provider = agent_cfg.get('provider', self.config.default_provider)
        model = agent_cfg.get('model')
        max_tokens = agent_cfg.get('max_tokens', 6000)
        temperature = agent_cfg.get('temperature', 0.2)
        client = build_client(self.config, provider)

        prompt = CONTENT_REWRITE_PROMPT.replace("{paper_text}", paper_text)
        prompt = prompt.replace("{extracted_json}", json.dumps(data, ensure_ascii=False, indent=2))

        response = await client.call(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=self.workflow.get('retry_attempts', 3),
            retry_delay=self.workflow.get('retry_delay', 2),
        )
        repaired = self._parse_json(response)
        if not repaired:
            return data
        return self._merge_preserve(data, repaired)

    async def _extract_keywords_with_llm(self, paper_text: str, max_keywords: int) -> List[str]:
        agent_cfg = self.config.agent_configs.get('keywords_extractor', {})
        if not agent_cfg:
            return []
        provider = agent_cfg.get('provider', self.config.default_provider)
        model = agent_cfg.get('model')
        max_tokens = agent_cfg.get('max_tokens', 1000)
        temperature = agent_cfg.get('temperature', 0.1)

        client = build_client(self.config, provider)
        prompt = KEYWORDS_PROMPT.replace("{text}", paper_text)
        prompt = prompt.replace("{max_keywords}", str(max_keywords))
        response = await client.call(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=self.workflow.get('retry_attempts', 3),
            retry_delay=self.workflow.get('retry_delay', 2),
        )
        parsed = self._parse_json(response)
        keywords = parsed.get('keywords', []) if isinstance(parsed, dict) else []
        if isinstance(keywords, list):
            return [k for k in keywords if isinstance(k, str) and k.strip()]
        return []

    def _split_sentences(self, text: str) -> List[tuple[int, int, str]]:
        sentences = []
        start = 0
        separators = {".", "!", "?", "\u3002", "\uff01", "\uff1f", "\n"}
        for idx, ch in enumerate(text):
            if ch in separators:
                end = idx + 1
                segment = text[start:end].strip()
                if segment:
                    sentences.append((start, end, segment))
                start = end
        if start < len(text):
            segment = text[start:].strip()
            if segment:
                sentences.append((start, len(text), segment))
        return sentences

    def _ref_id_in_group(self, ref_id: str, group: str) -> bool:
        ref_id = str(ref_id).strip()
        if not ref_id:
            return False
        tokens = group.replace(";", ",").replace("\uff1b", ",").split(",")
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if token == ref_id:
                return True
            # handle ranges like 4-9 or 4~9
            for sep in ("-", "~", "\u2013", "\u2014"):
                if sep in token:
                    parts = [p.strip() for p in token.split(sep) if p.strip()]
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and ref_id.isdigit():
                        if int(parts[0]) <= int(ref_id) <= int(parts[1]):
                            return True
        return False

    def _context_from_span(self, text: str, start: int, end: int, window: int) -> str:
        sentences = self._split_sentences(text)
        for idx, (s, e, seg) in enumerate(sentences):
            if start >= s and end <= e:
                lo = max(0, idx - window)
                hi = min(len(sentences), idx + window + 1)
                return " ".join([sentences[i][2] for i in range(lo, hi)])
        return ""

    def _find_citation_context(self, text: str, ref_id: str, citation_text: str, window: int) -> str:
        if not text:
            return ""
        import re
        # bracketed numeric citations
        for match in re.finditer(r"\[(.*?)\]", text):
            group = match.group(1)
            if self._ref_id_in_group(ref_id, group):
                return self._context_from_span(text, match.start(), match.end(), window)
        # parenthesized numeric citations
        for match in re.finditer(r"\(([^)]*?)\)", text):
            group = match.group(1)
            if self._ref_id_in_group(ref_id, group):
                return self._context_from_span(text, match.start(), match.end(), window)
        # superscript citations like ^1 or <sup>1</sup>
        for match in re.finditer(r"<sup>([^<]+)</sup>", text, flags=re.IGNORECASE):
            group = match.group(1)
            if self._ref_id_in_group(ref_id, group):
                return self._context_from_span(text, match.start(), match.end(), window)
        for match in re.finditer(r"\^\{?(\d[\d,;~\-\u2013\u2014 ]*)\}?", text):
            group = match.group(1)
            if self._ref_id_in_group(ref_id, group):
                return self._context_from_span(text, match.start(), match.end(), window)
        # fallback: try surname from citation
        if citation_text:
            author_year = re.search(r"([A-Za-z][A-Za-z'`-]+).*?(\d{4})", citation_text)
            if author_year:
                surname = author_year.group(1)
                year = author_year.group(2)
                pattern = re.compile(rf"{re.escape(surname)}[^\n\r]{{0,120}}{year}")
                match = pattern.search(text)
                if match:
                    return self._context_from_span(text, match.start(), match.end(), window)
            m = re.match(r"\s*([A-Za-z][A-Za-z'`-]*)", citation_text)
            if m:
                surname = m.group(1)
                idx = text.find(surname)
                if idx != -1:
                    return self._context_from_span(text, idx, idx + len(surname), window)
        return ""

    async def _call_citation_purpose_llm(self, citation: str, context: str) -> str:
        if not context:
            return ""
        agent_cfg = self.config.agent_configs.get('citation_purpose', {})
        if not agent_cfg:
            return ""
        provider = agent_cfg.get('provider', self.config.default_provider)
        model = agent_cfg.get('model')
        max_tokens = agent_cfg.get('max_tokens', 300)
        temperature = agent_cfg.get('temperature', 0.2)

        client = build_client(self.config, provider)
        prompt = CITATION_PURPOSE_PROMPT.replace("{citation}", citation or "")
        prompt = prompt.replace("{context}", context)
        response = await client.call(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=self.workflow.get('retry_attempts', 3),
            retry_delay=self.workflow.get('retry_delay', 2),
        )
        parsed = self._parse_json(response)
        if isinstance(parsed, dict):
            purpose = parsed.get('purpose')
            if isinstance(purpose, str):
                return purpose.strip()
        return ""

    async def _extract_citation_purposes(self, paper_text: str, multimedia: Dict[str, Any]) -> Dict[str, Any]:
        if not self.citation_purpose_cfg.get('enable', True):
            return multimedia
        references = multimedia.get('references', {}) if isinstance(multimedia, dict) else {}
        ref_list = references.get('reference_list', []) if isinstance(references, dict) else []
        if not isinstance(ref_list, list) or not ref_list:
            return multimedia

        max_refs = int(self.citation_purpose_cfg.get('max_refs', 50))
        window = int(self.citation_purpose_cfg.get('context_window', 1))
        for idx, item in enumerate(ref_list):
            if idx >= max_refs:
                break
            if not isinstance(item, dict):
                continue
            if isinstance(item.get('purpose'), str) and item.get('purpose').strip():
                continue
            ref_id = str(item.get('id') or "").strip()
            citation_text = item.get('citation') or ''
            context = self._find_citation_context(paper_text, ref_id, citation_text, window)
            try:
                purpose = await self._call_citation_purpose_llm(citation_text, context)
            except Exception as exc:
                logger.warning(f"Citation purpose LLM failed: {exc}")
                purpose = ""
            if purpose:
                item['purpose'] = purpose

        references['reference_list'] = ref_list
        multimedia['references'] = references
        return multimedia

    def _truncate_text(self, text: str) -> str:
        max_len = int(self.context.get('max_context_length', 500000))
        if len(text) <= max_len:
            return text
        return text[:max_len]

    async def _crossref_enrich(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not self.crossref_client:
            return metadata

        doi = metadata.get('doi')
        title = metadata.get('title')
        authors = [a.get('full_name', '') for a in metadata.get('authors', []) if isinstance(a, dict)]
        year = metadata.get('publication_year')

        crossref_data = await self.crossref_client.get_metadata(
            doi=doi,
            title=title,
            authors=authors,
            year=year,
            full_citation=None,
        )
        if not crossref_data:
            return metadata

        crossref_metadata = CrossrefDataExtractor.extract_complete_metadata(crossref_data)

        # Merge: Crossref fills missing fields, existing values kept
        for k, v in crossref_metadata.items():
            if k not in metadata or metadata.get(k) in (None, "", [], {}):
                metadata[k] = v
        return metadata

    async def _enrich_reference_dois(self, multimedia: Dict[str, Any]) -> Dict[str, Any]:
        if not self.crossref_client:
            return multimedia

        references = multimedia.get('references', {})
        ref_list = references.get('reference_list', [])
        if not isinstance(ref_list, list) or not ref_list:
            return multimedia

        for item in ref_list:
            if not isinstance(item, dict):
                continue
            if item.get('doi'):
                continue
            citation = item.get('citation') or ''
            if not citation:
                continue
            try:
                crossref_data = await self.crossref_client.get_metadata(full_citation=citation)
                if crossref_data and crossref_data.get('DOI'):
                    item['doi'] = crossref_data.get('DOI')
            except Exception:
                continue

        references['reference_list'] = ref_list
        multimedia['references'] = references
        return multimedia

    def _merge_preserve(self, original: Any, repaired: Any) -> Any:
        def _is_empty(val: Any) -> bool:
            return val is None or val == "" or val == {} or val == []

        if isinstance(original, dict) and isinstance(repaired, dict):
            merged = dict(repaired)
            for k, v in original.items():
                if k not in merged:
                    merged[k] = v
                    continue
                if _is_empty(merged[k]) and not _is_empty(v):
                    merged[k] = v
                    continue
                merged[k] = self._merge_preserve(v, merged[k])
            return merged
        return repaired if repaired is not None else original

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if not text:
            return []
        parts = re.split(r"[。！？!?；;\\n]+", text)
        return [p.strip() for p in parts if p.strip()]

    def _normalize_segment_map_fields(self, logic_chain: Dict[str, Any]) -> None:
        fields = [
            ("research_narrative", "background", "state_of_the_art_summary"),
            ("research_narrative", "problem_formulation", "research_objectives"),
            ("research_narrative", "methodology", "method_assumptions_and_limitations", "assumptions"),
            ("research_narrative", "methodology", "method_assumptions_and_limitations", "limitations"),
            ("research_narrative", "results_and_findings", "key_findings"),
            ("research_narrative", "discussion_and_conclusion", "limitations_of_this_work"),
            ("research_narrative", "discussion_and_conclusion", "future_work_suggestions"),
        ]

        for path in fields:
            node: Any = logic_chain
            for key in path:
                if not isinstance(node, dict):
                    node = None
                    break
                node = node.get(key)
            if not isinstance(node, dict):
                continue

            value = node.get("value")
            se = node.get("source_excerpt")
            text = None
            if isinstance(se, str):
                text = se
            elif isinstance(se, dict):
                if isinstance(se.get("text"), str) or se.get("text") is None:
                    text = se.get("text")

            segment_map = None
            if isinstance(se, dict):
                segment_map = se.get("segment_map")

            if not isinstance(segment_map, list):
                segment_map = []
                if isinstance(value, str) and value.strip():
                    sentences = self._split_sentences(value)
                    excerpt = text if isinstance(text, str) else ""
                    segment_map = [
                        {"value_segment": sentence, "excerpt": excerpt}
                        for sentence in sentences
                    ]
            else:
                # Ensure required keys exist
                fixed_segments = []
                for seg in segment_map:
                    if not isinstance(seg, dict):
                        continue
                    value_segment = seg.get("value_segment")
                    if not isinstance(value_segment, str) or not value_segment.strip():
                        continue
                    excerpt = seg.get("excerpt")
                    if not isinstance(excerpt, str):
                        excerpt = text if isinstance(text, str) else ""
                    fixed_segments.append({"value_segment": value_segment, "excerpt": excerpt})
                segment_map = fixed_segments

            node["source_excerpt"] = {
                "text": text,
                "segment_map": segment_map,
            }

    @staticmethod
    def _normalize_metadata_lists(logic_chain: Dict[str, Any]) -> None:
        meta = logic_chain.get("paper_metadata")
        if not isinstance(meta, dict):
            return
        identifiers = meta.get("identifiers")
        if isinstance(identifiers, dict):
            isbn = identifiers.get("isbn")
            if isinstance(isbn, list):
                identifiers["isbn"] = [str(x).strip() for x in isbn if isinstance(x, str) and str(x).strip()]
        categories = meta.get("categories")
        if isinstance(categories, dict):
            for key in ("categories", "subjects"):
                values = categories.get(key)
                if isinstance(values, list):
                    categories[key] = [str(x).strip() for x in values if isinstance(x, str) and str(x).strip()]
        funding = meta.get("funding")
        if isinstance(funding, list):
            meta["funding"] = [x for x in funding if isinstance(x, dict)]

    @staticmethod
    def _iter_logic_items(narrative: Dict[str, Any]):
        if not isinstance(narrative, dict):
            return
        list_sections = [
            ("background", "B"),
            ("state_of_art", "SA"),
            ("research_gaps", "G"),
            ("research_questions", "Q"),
            ("research_objectives", "O"),
            ("hypotheses", "H"),
        ]
        for section, prefix in list_sections:
            items = narrative.get(section)
            if isinstance(items, list):
                for idx, item in enumerate(items):
                    yield f"{section}[{idx}]", item, prefix
                    supports = item.get("supports") if isinstance(item, dict) else None
                    if isinstance(supports, list):
                        for sidx, support in enumerate(supports):
                            yield f"{section}[{idx}].supports[{sidx}]", support, f"{prefix}S"

        hub_sections = [
            ("background_hub", "BH"),
            ("state_hub", "SH"),
            ("gap_hub", "GH"),
            ("objective_hub", "OH"),
            ("hypothesis_hub", "HH"),
        ]
        for section, prefix in hub_sections:
            item = narrative.get(section)
            if isinstance(item, dict):
                yield section, item, prefix
                supports = item.get("supports")
                if isinstance(supports, list):
                    for sidx, support in enumerate(supports):
                        yield f"{section}.supports[{sidx}]", support, f"{prefix}S"

        for block_name, prefix_main, prefix_support in (
            ("methods", "M", "MS"),
            ("results", "R", "RS"),
            ("conclusions", "C", "CS"),
        ):
            block = narrative.get(block_name)
            if not isinstance(block, dict):
                continue
            main = block.get("main")
            if isinstance(main, dict):
                yield f"{block_name}.main", main, prefix_main
                supports = main.get("supports")
                if isinstance(supports, list):
                    for sidx, support in enumerate(supports):
                        yield f"{block_name}.main.supports[{sidx}]", support, f"{prefix_main}S"
            supports = block.get("supports")
            if isinstance(supports, list):
                for idx, item in enumerate(supports):
                    yield f"{block_name}.supports[{idx}]", item, prefix_support
                    nested_supports = item.get("supports") if isinstance(item, dict) else None
                    if isinstance(nested_supports, list):
                        for sidx, support in enumerate(nested_supports):
                            yield f"{block_name}.supports[{idx}].supports[{sidx}]", support, f"{prefix_support}S"

    def _normalize_logic_item(
        self,
        item: Dict[str, Any],
        prefix: str,
        counter: int,
        segments_by_id: Dict[str, Dict[str, Any]],
        used_ids: set,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        value = item.get("value")
        if not isinstance(value, str) or not value.strip():
            return None
        evidence_ids = item.get("evidence_segment_ids")
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        filtered_ids = []
        seen = set()
        for sid in evidence_ids:
            if not isinstance(sid, str):
                continue
            if sid in segments_by_id and sid not in seen:
                seen.add(sid)
                filtered_ids.append(sid)
        if not filtered_ids:
            return None

        node_id = item.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            node_id = f"{prefix}{counter}"
        node_id = node_id.strip()
        if node_id in used_ids:
            node_id = f"{node_id}_{counter}"
        used_ids.add(node_id)

        citations = item.get("citations")
        if not isinstance(citations, list):
            citations = []
        normalized_citations = []
        for cite in citations:
            if not isinstance(cite, dict):
                continue
            citation_id = cite.get("citation_id")
            if citation_id is not None and not isinstance(citation_id, str):
                citation_id = str(citation_id)
            citation_text = cite.get("citation_text")
            if citation_text is not None and not isinstance(citation_text, str):
                citation_text = str(citation_text)
            purpose = cite.get("purpose")
            if not isinstance(purpose, str):
                purpose = ""
            normalized_citations.append(
                {
                    "citation_id": citation_id,
                    "citation_text": citation_text,
                    "purpose": purpose,
                }
            )

        item = dict(item)
        item["node_id"] = node_id
        item["value"] = value.strip()
        item["evidence_segment_ids"] = filtered_ids
        item["citations"] = normalized_citations
        return item

    def _normalize_research_narrative(self, logic_chain: Dict[str, Any], segments_by_id: Dict[str, Dict[str, Any]]) -> None:
        narrative = logic_chain.get("research_narrative")
        if not isinstance(narrative, dict):
            return
        narrative = self._ensure_research_narrative_sections(narrative)
        used_ids: set = set()
        counters = {
            "B": 1,
            "BS": 1,
            "SA": 1,
            "SAS": 1,
            "G": 1,
            "GS": 1,
            "Q": 1,
            "QS": 1,
            "O": 1,
            "OS": 1,
            "H": 1,
            "HS": 1,
            "BH": 1,
            "BHS": 1,
            "SH": 1,
            "SHS": 1,
            "GH": 1,
            "GHS": 1,
            "OH": 1,
            "OHS": 1,
            "HH": 1,
            "HHS": 1,
            "M": 1,
            "MS": 1,
            "MSS": 1,
            "R": 1,
            "RS": 1,
            "RSS": 1,
            "C": 1,
            "CS": 1,
            "CSS": 1,
        }

        # Normalize list sections (with optional supports)
        list_sections = (
            ("background", "B", "BS"),
            ("state_of_art", "SA", "SAS"),
            ("research_gaps", "G", "GS"),
            ("research_questions", "Q", "QS"),
            ("research_objectives", "O", "OS"),
            ("hypotheses", "H", "HS"),
        )
        for section, prefix, support_prefix in list_sections:
            items = narrative.get(section, [])
            normalized = []
            if isinstance(items, list):
                for item in items:
                    normalized_item = self._normalize_logic_item(
                        item,
                        prefix,
                        counters[prefix],
                        segments_by_id,
                        used_ids,
                    )
                    if normalized_item:
                        counters[prefix] += 1
                        supports = item.get("supports") if isinstance(item, dict) else None
                        if isinstance(supports, list):
                            normalized_supports = []
                            for support in supports:
                                normalized_support = self._normalize_logic_item(
                                    support,
                                    support_prefix,
                                    counters[support_prefix],
                                    segments_by_id,
                                    used_ids,
                                )
                                if normalized_support:
                                    normalized_supports.append(normalized_support)
                                    counters[support_prefix] += 1
                            if normalized_supports:
                                normalized_item["supports"] = normalized_supports
                        normalized.append(normalized_item)
            narrative[section] = normalized

        # Ensure research_objectives exist; fallback to research_questions when missing
        if not narrative.get("research_objectives") and narrative.get("research_questions"):
            fallback_objectives: List[Dict[str, Any]] = []
            for q_item in narrative.get("research_questions", []):
                if not isinstance(q_item, dict):
                    continue
                obj_item = dict(q_item)
                obj_node = f"O{counters['O']}"
                while obj_node in used_ids:
                    counters["O"] += 1
                    obj_node = f"O{counters['O']}"
                obj_item["node_id"] = obj_node
                used_ids.add(obj_node)
                counters["O"] += 1
                fallback_objectives.append(obj_item)
            narrative["research_objectives"] = fallback_objectives

        def _build_hub_from_list(list_key: str, hub_key: str, label: str, prefix: str) -> None:
            if narrative.get(hub_key) is not None:
                return
            items = narrative.get(list_key)
            if not isinstance(items, list) or not items:
                return
            first = items[0] if isinstance(items[0], dict) else None
            if not first:
                return
            evid = first.get("evidence_segment_ids") if isinstance(first.get("evidence_segment_ids"), list) else []
            if not evid:
                return
            narrative[hub_key] = {
                "node_id": f"{prefix}{counters[prefix]}",
                "value": label,
                "evidence_segment_ids": evid[:3],
                "citations": [],
            }
            counters[prefix] += 1

        _build_hub_from_list("background", "background_hub", "研究背景汇总节点", "BH")
        _build_hub_from_list("state_of_art", "state_hub", "研究现状汇总节点", "SH")
        _build_hub_from_list("research_gaps", "gap_hub", "研究缺口汇总节点", "GH")
        _build_hub_from_list("research_objectives", "objective_hub", "研究目的汇总节点", "OH")
        _build_hub_from_list("hypotheses", "hypothesis_hub", "研究假设汇总节点", "HH")

        # Normalize hub nodes (with optional supports)
        hub_sections = (
            ("background_hub", "BH", "BHS"),
            ("state_hub", "SH", "SHS"),
            ("gap_hub", "GH", "GHS"),
            ("objective_hub", "OH", "OHS"),
            ("hypothesis_hub", "HH", "HHS"),
        )
        for section, prefix, support_prefix in hub_sections:
            item = narrative.get(section)
            normalized_item = None
            if isinstance(item, dict):
                normalized_item = self._normalize_logic_item(
                    item,
                    prefix,
                    counters[prefix],
                    segments_by_id,
                    used_ids,
                )
                if normalized_item:
                    counters[prefix] += 1
                    supports = item.get("supports")
                    if isinstance(supports, list):
                        normalized_supports = []
                        for support in supports:
                            normalized_support = self._normalize_logic_item(
                                support,
                                support_prefix,
                                counters[support_prefix],
                                segments_by_id,
                                used_ids,
                            )
                            if normalized_support:
                                normalized_supports.append(normalized_support)
                                counters[support_prefix] += 1
                        if normalized_supports:
                            normalized_item["supports"] = normalized_supports
            narrative[section] = normalized_item

        # Normalize main/support blocks
        for block_name, prefix_main, prefix_support in (
            ("methods", "M", "MS"),
            ("results", "R", "RS"),
            ("conclusions", "C", "CS"),
        ):
            block = narrative.get(block_name, {})
            if not isinstance(block, dict):
                block = {"main": None, "supports": []}
            main = block.get("main")
            normalized_main = None
            if isinstance(main, dict):
                normalized_main = self._normalize_logic_item(
                    main,
                    prefix_main,
                    counters[prefix_main],
                    segments_by_id,
                    used_ids,
                )
                if normalized_main:
                    counters[prefix_main] += 1
                    main_supports = main.get("supports")
                    if isinstance(main_supports, list):
                        normalized_supports = []
                        for support in main_supports:
                            normalized_support = self._normalize_logic_item(
                                support,
                                f"{prefix_main}S",
                                counters[f"{prefix_main}S"],
                                segments_by_id,
                                used_ids,
                            )
                            if normalized_support:
                                normalized_supports.append(normalized_support)
                                counters[f"{prefix_main}S"] += 1
                        if normalized_supports:
                            normalized_main["supports"] = normalized_supports
            supports = block.get("supports")
            normalized_supports = []
            if isinstance(supports, list):
                for item in supports:
                    normalized_item = self._normalize_logic_item(
                        item,
                        prefix_support,
                        counters[prefix_support],
                        segments_by_id,
                        used_ids,
                    )
                    if normalized_item:
                        normalized_supports.append(normalized_item)
                        counters[prefix_support] += 1
                        nested_supports = item.get("supports") if isinstance(item, dict) else None
                        if isinstance(nested_supports, list):
                            nested_normalized = []
                            nested_prefix = f"{prefix_support}S"
                            for support in nested_supports:
                                normalized_nested = self._normalize_logic_item(
                                    support,
                                    nested_prefix,
                                    counters[nested_prefix],
                                    segments_by_id,
                                    used_ids,
                                )
                                if normalized_nested:
                                    nested_normalized.append(normalized_nested)
                                    counters[nested_prefix] += 1
                            if nested_normalized:
                                normalized_item["supports"] = nested_normalized
            narrative[block_name] = {"main": normalized_main, "supports": normalized_supports}

        self._normalize_logic_chains(narrative, used_ids)
        logic_chain["research_narrative"] = narrative

    async def _link_state_gap_objective(self, narrative: Dict[str, Any]) -> List[Dict[str, str]]:
        states = narrative.get("state_of_art") if isinstance(narrative.get("state_of_art"), list) else []
        gaps = narrative.get("research_gaps") if isinstance(narrative.get("research_gaps"), list) else []
        objectives = narrative.get("research_objectives") if isinstance(narrative.get("research_objectives"), list) else []

        def _pack(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
            packed: List[Dict[str, str]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                node_id = item.get("node_id")
                value = item.get("value")
                if isinstance(node_id, str) and isinstance(value, str) and value.strip():
                    packed.append({"id": node_id, "text": value.strip()})
            return packed

        payload = json.dumps(
            {
                "states": _pack(states),
                "gaps": _pack(gaps),
                "objectives": _pack(objectives),
            },
            ensure_ascii=False,
            indent=2,
        )
        if payload.strip() == "{}":
            return []
        try:
            parsed = await self._call_agent(
                "research_narrative_selector",
                STATE_GAP_OBJECTIVE_LINK_PROMPT,
                payload,
                strict_json=True,
            )
        except Exception as exc:
            logger.warning(f"State-gap-objective linking failed: {exc}")
            parsed = {}

        relations: List[Dict[str, str]] = []
        if isinstance(parsed, dict):
            state_gap = parsed.get("state_gap_links", [])
            if isinstance(state_gap, list):
                for link in state_gap:
                    if not isinstance(link, dict):
                        continue
                    sid = link.get("state_id")
                    gid = link.get("gap_id")
                    if isinstance(sid, str) and isinstance(gid, str):
                        relations.append({"from_id": sid, "to_id": gid, "relation_type": "LEADS_TO_GAP"})
            gap_obj = parsed.get("gap_objective_links", [])
            if isinstance(gap_obj, list):
                for link in gap_obj:
                    if not isinstance(link, dict):
                        continue
                    gid = link.get("gap_id")
                    oid = link.get("objective_id")
                    if isinstance(gid, str) and isinstance(oid, str):
                        relations.append({"from_id": gid, "to_id": oid, "relation_type": "MOTIVATES_OBJECTIVE"})
        return relations

    async def _augment_node_relations(self, logic_chain: Dict[str, Any]) -> None:
        narrative = logic_chain.get("research_narrative")
        if not isinstance(narrative, dict):
            return
        relations: List[Dict[str, str]] = []

        def _hub_links(list_key: str, hub_key: str) -> None:
            items = narrative.get(list_key)
            hub = narrative.get(hub_key)
            if not isinstance(items, list) or not isinstance(hub, dict):
                return
            hub_id = hub.get("node_id")
            if not isinstance(hub_id, str):
                return
            for item in items:
                if not isinstance(item, dict):
                    continue
                nid = item.get("node_id")
                if isinstance(nid, str):
                    relations.append({"from_id": nid, "to_id": hub_id, "relation_type": "AGGREGATES_TO"})

        _hub_links("background", "background_hub")
        _hub_links("state_of_art", "state_hub")
        _hub_links("research_gaps", "gap_hub")
        _hub_links("research_objectives", "objective_hub")
        _hub_links("hypotheses", "hypothesis_hub")

        relations.extend(await self._link_state_gap_objective(narrative))

        # Deduplicate
        dedup = []
        seen = set()
        for rel in relations:
            key = (rel.get("from_id"), rel.get("to_id"), rel.get("relation_type"))
            if key in seen or not all(key):
                continue
            seen.add(key)
            dedup.append(rel)
        narrative["node_relations"] = dedup

    def _normalize_logic_chains(self, narrative: Dict[str, Any], node_ids: set) -> None:
        # Always rebuild canonical chains to ensure main-only steps and multi-chain split
        narrative["logic_chains"] = self._build_default_chains(narrative)

    def _build_default_chains(self, narrative: Dict[str, Any]) -> List[Dict[str, Any]]:
        def _node_id(obj: Any) -> Optional[str]:
            if isinstance(obj, dict):
                nid = obj.get("node_id")
                if isinstance(nid, str):
                    return nid
            return None

        def _list_node_ids(items: Any) -> List[str]:
            ids: List[str] = []
            if isinstance(items, list):
                for item in items:
                    nid = _node_id(item)
                    if nid:
                        ids.append(nid)
            return ids

        background_hub = _node_id(narrative.get("background_hub"))
        state_hub = _node_id(narrative.get("state_hub"))
        gap_hub = _node_id(narrative.get("gap_hub"))
        objective_hub = _node_id(narrative.get("objective_hub"))
        hypothesis_hub = _node_id(narrative.get("hypothesis_hub"))

        method_id = _node_id(narrative.get("methods", {}).get("main") if isinstance(narrative.get("methods"), dict) else None)
        result_id = _node_id(narrative.get("results", {}).get("main") if isinstance(narrative.get("results"), dict) else None)
        conclusion_id = _node_id(narrative.get("conclusions", {}).get("main") if isinstance(narrative.get("conclusions"), dict) else None)

        base_steps = [
            background_hub,
            state_hub,
            gap_hub,
            objective_hub,
            hypothesis_hub,
            method_id,
            result_id,
            conclusion_id,
        ]
        base_steps = [sid for sid in base_steps if isinstance(sid, str)]

        objective_items = narrative.get("research_objectives") if isinstance(narrative.get("research_objectives"), list) else []
        question_ids = _list_node_ids(narrative.get("research_questions"))
        hypothesis_ids = _list_node_ids(narrative.get("hypotheses"))

        chains: List[Dict[str, Any]] = []
        if objective_items:
            chain_idx = 1
            for obj in objective_items:
                obj_id = _node_id(obj)
                if not obj_id:
                    continue
                chains.append(
                    {
                        "chain_id": f"C{chain_idx}",
                        "objective_ids": [obj_id],
                        "question_ids": question_ids,
                        "hypothesis_ids": hypothesis_ids,
                        "steps": base_steps,
                    }
                )
                chain_idx += 1
        else:
            chains.append(
                {
                    "chain_id": "C1",
                    "objective_ids": [],
                    "question_ids": question_ids,
                    "hypothesis_ids": hypothesis_ids,
                    "steps": base_steps,
                }
            )
        return chains

    def _collect_evidence_ids_from_narrative(self, narrative: Dict[str, Any]) -> List[str]:
        evidence_ids: List[str] = []
        if not isinstance(narrative, dict):
            return evidence_ids
        for _path, item, _prefix in self._iter_logic_items(narrative):
            if not isinstance(item, dict):
                continue
            ids = item.get("evidence_segment_ids")
            if not isinstance(ids, list):
                continue
            for sid in ids:
                if isinstance(sid, str):
                    evidence_ids.append(sid)
        return evidence_ids

    def _build_coverage_report(
        self,
        narrative: Dict[str, Any],
        segments_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        used_ids = self._collect_evidence_ids_from_narrative(narrative)
        used_set = set(used_ids)
        total_segments = len(segments_by_id)

        section_totals: Dict[str, int] = {}
        section_used: Dict[str, int] = {}
        for sid, seg in segments_by_id.items():
            section = seg.get("section") if isinstance(seg, dict) else None
            if not isinstance(section, str) or not section:
                section = "unknown"
            section_totals[section] = section_totals.get(section, 0) + 1
            if sid in used_set:
                section_used[section] = section_used.get(section, 0) + 1

        section_stats = {}
        for section, total in section_totals.items():
            used = section_used.get(section, 0)
            section_stats[section] = {
                "total": total,
                "used": used,
                "coverage": round(used / total, 4) if total > 0 else 0.0,
            }

        unused_ids = [sid for sid in segments_by_id.keys() if sid not in used_set]
        sample_limit = 200
        report = {
            "total_segments": total_segments,
            "used_segments": len(used_set),
            "coverage": round(len(used_set) / total_segments, 4) if total_segments > 0 else 0.0,
            "section_stats": section_stats,
            "missing_sections": [sec for sec, stats in section_stats.items() if stats["total"] > 0 and stats["used"] == 0],
            "unused_segment_ids_sample": unused_ids[:sample_limit],
            "unused_segment_count": len(unused_ids),
        }
        return report

    async def _enrich_logic_node_citations(
        self,
        logic_chain: Dict[str, Any],
        segments_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        enable_purpose = bool(self.citation_purpose_cfg.get("enable", True))
        narrative = logic_chain.get("research_narrative")
        if not isinstance(narrative, dict):
            return logic_chain

        references = (
            logic_chain.get("multimedia_content", {})
            .get("references", {})
            .get("reference_list", [])
        )
        ref_by_id: Dict[str, Dict[str, Any]] = {}
        ref_by_text: Dict[str, Dict[str, Any]] = {}
        if isinstance(references, list):
            for ref in references:
                if not isinstance(ref, dict):
                    continue
                ref_id = ref.get("id")
                if ref_id is not None:
                    ref_by_id[str(ref_id).strip()] = ref
                citation_text = ref.get("citation")
                if isinstance(citation_text, str) and citation_text.strip():
                    key = " ".join(citation_text.lower().split())
                    ref_by_text[key] = ref

        async def _fill_purpose(citation_text: str, context: str) -> str:
            try:
                return await self._call_citation_purpose_llm(citation_text, context)
            except Exception as exc:
                logger.warning(f"Logic node citation purpose failed: {exc}")
                return ""

        tasks = []
        task_slots = []
        max_refs = int(self.citation_purpose_cfg.get("max_refs", 50))
        context_limit = int(self.context.get("citation_context_chars", 2000))

        for path, item, _prefix in self._iter_logic_items(narrative):
            if not isinstance(item, dict):
                continue
            evidence_ids = item.get("evidence_segment_ids")
            if not isinstance(evidence_ids, list):
                evidence_ids = []
            segment_texts = []
            for sid in evidence_ids:
                seg = segments_by_id.get(sid)
                if not seg:
                    continue
                segment_texts.append(self._clean_segment_text(seg.get("text", "")))
            context = "\n".join(segment_texts)
            if len(context) > context_limit:
                context = context[:context_limit]

            citations = item.get("citations")
            if not isinstance(citations, list):
                citations = []
            # If citations empty (or missing ids), derive from segments
            if not citations or all(
                isinstance(c, dict) and not c.get("citation_id") and not c.get("citation_text")
                for c in citations
            ):
                derived_ids = []
                for sid in evidence_ids:
                    seg = segments_by_id.get(sid)
                    if not seg:
                        continue
                    for cid in seg.get("citations", []):
                        if cid not in derived_ids:
                            derived_ids.append(cid)
                citations = [{"citation_id": cid, "citation_text": None, "purpose": ""} for cid in derived_ids]
                item["citations"] = citations

            for cite in citations:
                if not isinstance(cite, dict):
                    continue
                if len(tasks) >= max_refs:
                    break
                citation_id = cite.get("citation_id")
                citation_text = cite.get("citation_text")
                if not citation_text and citation_id:
                    ref = ref_by_id.get(str(citation_id))
                    if ref and isinstance(ref.get("citation"), str):
                        citation_text = ref.get("citation")
                        cite["citation_text"] = citation_text
                if not citation_text and citation_id is None:
                    citation_text = ""
                if not isinstance(cite.get("purpose"), str) or not cite.get("purpose"):
                    tasks.append(_fill_purpose(citation_text or "", context))
                    task_slots.append((cite, "purpose"))

        if enable_purpose and tasks:
            results = await asyncio.gather(*tasks)
            for (cite, key), purpose in zip(task_slots, results):
                if isinstance(purpose, str):
                    cite[key] = purpose.strip()
        return logic_chain

    async def extract_file(self, file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        file_text = Path(file_path).read_text(encoding=self.config.get('file_processing', {}).get('text_encoding', 'utf-8'))
        paper_text = self._truncate_text(file_text)
        logger.info("Extraction mode: narrative_only | text_chars=%s", len(paper_text))
        evidence_segments = self._build_evidence_segments(file_text)
        stage_times: Dict[str, float] = {}
        extract_start = time.perf_counter()

        def _is_empty(value: Any) -> bool:
            return value in (None, "", [], {})

        def _empty_dict(obj: Any) -> bool:
            if not isinstance(obj, dict):
                return True
            return all(_is_empty(v) for v in obj.values())

        logic_chain: Dict[str, Any] = {}

        t = time.perf_counter()
        metadata = await self._extract_metadata(paper_text)
        metadata = self._prune_metadata_unrelated(metadata)
        stage_times['metadata'] = time.perf_counter() - t

        t = time.perf_counter()
        narrative = await self._extract_research_narrative(paper_text, evidence_segments)
        stage_times['research_narrative'] = time.perf_counter() - t

        t = time.perf_counter()
        multimedia = await self._extract_multimedia(paper_text)
        stage_times['multimedia'] = time.perf_counter() - t

        logic_chain = {
            'paper_metadata': metadata,
            'research_narrative': narrative,
            'multimedia_content': multimedia,
        }
        stage_times['extract'] = time.perf_counter() - extract_start

        rn = logic_chain.get('research_narrative', {}) if isinstance(logic_chain.get('research_narrative'), dict) else {}
        if 'conclusion' in rn and not rn.get('discussion_and_conclusion'):
            rn['discussion_and_conclusion'] = rn.pop('conclusion')
            logic_chain['research_narrative'] = rn

        # Enrich reference DOIs via Crossref when missing
        t = time.perf_counter()
        multimedia = logic_chain.get('multimedia_content', {})
        multimedia = await self._enrich_reference_dois(multimedia)
        # Add LLM-based citation purpose summaries
        multimedia = await self._extract_citation_purposes(paper_text, multimedia)
        logic_chain['multimedia_content'] = multimedia
        stage_times['crossref'] = time.perf_counter() - t

        iterative_enabled = bool(self.iterative_cfg.get('enable', False))
        max_rounds = int(self.iterative_cfg.get('max_rounds', 3))
        stop_on_no_change = bool(self.iterative_cfg.get('stop_on_no_change', True))

        if iterative_enabled:
            t = time.perf_counter()
            narrative = logic_chain.get('research_narrative', {})
            threshold = int(self.quality_cfg.get('threshold', 75))
            for round_idx in range(max_rounds):
                score = await self._compute_quality_score({'research_narrative': narrative}, paper_text)
                logger.info(
                    "Iterative narrative refine round %s/%s: score=%s",
                    round_idx + 1,
                    max_rounds,
                    score,
                )
                if score >= threshold:
                    break
                before = json.dumps(narrative, ensure_ascii=False, sort_keys=True)
                narrative = await self._refine_research_narrative(narrative, paper_text)
                after = json.dumps(narrative, ensure_ascii=False, sort_keys=True)
                if stop_on_no_change and after == before:
                    logger.info("Iterative narrative refine stopped: no changes in round %s", round_idx + 1)
                    break
            logic_chain['research_narrative'] = narrative
            stage_times['iterative_refine'] = time.perf_counter() - t
        else:
            # Quality scoring + refinement
            if self.quality_cfg.get('enable', True):
                t = time.perf_counter()
                final_score = await self._compute_quality_score(logic_chain, paper_text)
                threshold = int(self.quality_cfg.get('threshold', 75))
                if final_score < threshold:
                    rounds = int(self.quality_cfg.get('refine_rounds', 1))
                    for _ in range(max(1, rounds)):
                        logic_chain = await self._refine_with_llm(logic_chain, paper_text)
                stage_times['quality'] = time.perf_counter() - t

            # Optional schema repair
            if self.config.get('repair', {}).get('enable_schema_repair', True):
                t = time.perf_counter()
                logic_chain = await self._repair_json_with_llm(logic_chain, paper_text)
                stage_times['repair'] = time.perf_counter() - t

        # Normalize narrative and metadata fields
        segments_by_id = {seg.get("id"): seg for seg in evidence_segments if isinstance(seg, dict) and seg.get("id")}
        self._normalize_research_narrative(logic_chain, segments_by_id)
        self._normalize_metadata_lists(logic_chain)
        await self._augment_node_relations(logic_chain)
        logic_chain = await self._enrich_logic_node_citations(logic_chain, segments_by_id)
        coverage_report = self._build_coverage_report(logic_chain.get("research_narrative", {}), segments_by_id)

        # Validate
        t = time.perf_counter()
        errors = self.validator.validate(logic_chain)
        if errors:
            logger.warning("Schema validation errors:\n" + "\n".join(errors[:20]))
        stage_times['validate'] = time.perf_counter() - t

        # Write output
        t = time.perf_counter()
        if output_path:
            out_path = Path(output_path)
        else:
            out_dir = Path(file_path).parent / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{Path(file_path).stem}_logic_chain.json"

        out_path.write_text(
            json.dumps(logic_chain, ensure_ascii=self.output_cfg.get('ensure_ascii', False), indent=self.output_cfg.get('indent', 2)),
            encoding='utf-8'
        )
        # Write evidence segments sidecar file (full coverage of original text)
        try:
            sidecar_path = out_path.with_name(f"{out_path.stem}_evidence_segments.json")
            sidecar_payload = {
                "source": str(file_path),
                "segment_count": len(evidence_segments),
                "segments": evidence_segments,
            }
            sidecar_path.write_text(
                json.dumps(sidecar_payload, ensure_ascii=self.output_cfg.get('ensure_ascii', False), indent=self.output_cfg.get('indent', 2)),
                encoding='utf-8'
            )
        except Exception as exc:
            logger.warning(f"Evidence sidecar write failed: {exc}")
        # Write coverage report sidecar
        try:
            coverage_path = out_path.with_name(f"{out_path.stem}_coverage.json")
            coverage_payload = {
                "source": str(file_path),
                "output": str(out_path),
                "coverage": coverage_report,
            }
            coverage_path.write_text(
                json.dumps(coverage_payload, ensure_ascii=self.output_cfg.get('ensure_ascii', False), indent=self.output_cfg.get('indent', 2)),
                encoding='utf-8'
            )
        except Exception as exc:
            logger.warning(f"Coverage sidecar write failed: {exc}")
        stage_times['output'] = time.perf_counter() - t

        refs = logic_chain.get('multimedia_content', {}).get('references', {}).get('reference_list', [])
        images = logic_chain.get('multimedia_content', {}).get('images', {})
        formulas = logic_chain.get('multimedia_content', {}).get('formulas', {}).get('formula_list', [])
        keywords = logic_chain.get('paper_metadata', {}).get('keywords', [])
        image_count = 0
        if isinstance(images, dict):
            for items in images.values():
                if isinstance(items, list):
                    image_count += len(items)

        logger.info(
            "Stage timings (s): %s",
            ", ".join([f"{k}={v:.1f}" for k, v in stage_times.items()])
        )
        logger.info(
            "Output stats: references=%s images=%s formulas=%s keywords=%s",
            len(refs) if isinstance(refs, list) else 0,
            image_count,
            len(formulas) if isinstance(formulas, list) else 0,
            len(keywords) if isinstance(keywords, list) else 0,
        )

        if self.neo4j_cfg.get('enable', False):
            try:
                from storage.exporters.neo4j_exporter import convert_paper_to_neo4j
                from storage.neo4j.neo4j_utils import clear_neo4j_database, get_graph_stats, get_neo4j_connection

                def _normalize_doi(value: Optional[str]) -> Optional[str]:
                    if not value:
                        return None
                    doi = str(value).strip()
                    if not doi:
                        return None
                    lowered = doi.lower()
                    if "doi.org/" in lowered:
                        doi = doi.split("doi.org/")[-1]
                    return doi.strip().lower() or None

                def _citation_hash(value: str) -> str:
                    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

                def _paper_key_from_metadata(meta: Dict[str, Any]) -> Optional[Dict[str, str]]:
                    doi = _normalize_doi(meta.get("doi") or meta.get("url"))
                    if doi:
                        return {"key_type": "doi", "key_value": doi}

                    title = str(meta.get("title") or "").strip()
                    year = str(meta.get("publication_year") or "").strip()
                    authors = meta.get("authors") if isinstance(meta.get("authors"), list) else []
                    author_names = []
                    for author in authors:
                        if isinstance(author, dict):
                            name = author.get("full_name") or author.get("family_name") or author.get("given_name")
                            if name:
                                author_names.append(str(name))
                    seed_parts = [part for part in [title, year, "|".join(author_names[:3])] if part]
                    seed = "|".join(seed_parts)
                    if not seed:
                        return None
                    return {"key_type": "citation_hash", "key_value": _citation_hash(seed)}

                def _neo4j_integrity_check(meta: Dict[str, Any]) -> Dict[str, Any]:
                    key = _paper_key_from_metadata(meta)
                    if not key:
                        return {"found": False, "reason": "missing_key"}
                    key_type = key["key_type"]
                    key_value = key["key_value"]
                    if key_type not in ("doi", "citation_hash"):
                        return {"found": False, "reason": "invalid_key"}
                    conn = get_neo4j_connection()
                    rows = conn.execute_query(
                        f"""
                        MATCH (p:Paper {{{key_type}: $key}})
                        OPTIONAL MATCH (p)-[:HAS_MULTIMEDIA_CONTENT]->(mm)
                        OPTIONAL MATCH (p)-[:CITES]->(c)
                        RETURN count(mm) as mm, count(c) as cites
                        """,
                        {"key": key_value},
                    )
                    if not rows:
                        return {"found": False, "key_type": key_type, "key_value": key_value}
                    row = rows[0]
                    return {
                        "found": True,
                        "key_type": key_type,
                        "key_value": key_value,
                        "has_multimedia_content": bool(row.get("mm")),
                        "cites": int(row.get("cites") or 0),
                    }

                if self.neo4j_cfg.get('clear_before_import', False) and not self._neo4j_cleared:
                    clear_neo4j_database()
                    self._neo4j_cleared = True
                import_ok = convert_paper_to_neo4j(
                    logic_chain,
                    self.neo4j_cfg.get('mapping_path', 'config/neo4j_mapping.json')
                )
                if not import_ok:
                    logger.warning("Neo4j export failed for %s", file_path)

                try:
                    integrity = _neo4j_integrity_check(logic_chain.get("paper_metadata", {}))
                    if integrity.get("found") and (not integrity.get("has_multimedia_content")):
                        logger.warning(
                            "Neo4j integrity check failed for %s: %s",
                            file_path,
                            integrity,
                        )
                    report = {
                        "source": str(file_path),
                        "output": str(out_path),
                        "stats": get_graph_stats(),
                        "neo4j_import": {
                            "success": import_ok,
                            "integrity": integrity,
                        },
                        "coverage": coverage_report,
                    }
                    report_path = out_path.with_name(f"{out_path.stem}_kg_report.json")
                    report_path.write_text(
                        json.dumps(report, ensure_ascii=self.output_cfg.get('ensure_ascii', False), indent=2),
                        encoding='utf-8'
                    )
                except Exception as exc:
                    logger.warning(f"Neo4j report failed: {exc}")
            except Exception as exc:
                logger.warning(f"Neo4j export failed: {exc}")
        return logic_chain

    def extract_file_sync(self, file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        if self.workflow.get('enable_async', True):
            return asyncio.run(self.extract_file(file_path, output_path))
        return asyncio.run(self.extract_file(file_path, output_path))
