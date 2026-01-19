"""Main extraction pipeline."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import Config
from core.schema import SchemaLoader
from core.validation import SchemaValidator
from core.scoring import rule_score
from core.crossref import CrossrefClient, CrossrefDataExtractor
from core.keywords import extract_keywords
from llm.prompts import (
    METADATA_PROMPT_BASE,
    BACKGROUND_PROMPT_BASE,
    METHODOLOGY_PROMPT_BASE,
    RESULTS_PROMPT_BASE,
    MULTIMEDIA_CONTENT_PROMPT_BASE,
    JSON_SCHEMA_REPAIR_PROMPT,
    QUALITY_RATER_PROMPT,
    CONTENT_REWRITE_PROMPT,
    KEYWORDS_PROMPT,
    CITATION_PURPOSE_PROMPT,
    FULL_EXTRACT_PROMPT_BASE,
    RESEARCH_NARRATIVE_PROMPT_BASE,
    RESEARCH_NARRATIVE_EVIDENCE_PROMPT,
    RESEARCH_NARRATIVE_SYNTH_PROMPT,
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
        self.extraction_mode = self.workflow.get('extraction_mode', 'multi')
        self.enable_targeted_fallback = self.workflow.get('enable_targeted_fallback', True)
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
    def _split_paragraphs(text: str) -> List[str]:
        parts = re.split(r"\n\\s*\n", text)
        return [p.strip() for p in parts if p.strip()]

    def _build_section_contexts(self, text: str) -> Dict[str, str]:
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return {
                "background": "",
                "problem_formulation": "",
                "methodology": "",
                "results_and_findings": "",
                "discussion_and_conclusion": "",
            }

        section_keywords = {
            "background": ["introduction", "background", "related work", "literature", "previous work", "引言", "背景", "相关工作", "文献综述"],
            "problem_formulation": ["problem", "objective", "aim", "goal", "question", "研究问题", "研究目标", "研究空白", "目的"],
            "methodology": ["method", "methods", "methodology", "experiment", "data", "materials", "simulation", "方法", "实验", "数据", "材料", "数值", "模拟"],
            "results_and_findings": ["results", "findings", "analysis", "observed", "结果", "发现", "分析", "实验结果"],
            "discussion_and_conclusion": ["discussion", "conclusion", "limitations", "future work", "讨论", "结论", "局限", "未来工作"],
        }

        def match_paragraph(paragraph: str, keywords: List[str]) -> bool:
            lower = paragraph.lower()
            for kw in keywords:
                if kw.lower() in lower:
                    return True
            return False

        contexts: Dict[str, List[str]] = {k: [] for k in section_keywords}
        for para in paragraphs:
            matched = False
            for section, keywords in section_keywords.items():
                if match_paragraph(para, keywords):
                    contexts[section].append(para)
                    matched = True
            if not matched:
                continue

        # fallback: if a section is empty, use early/middle/late paragraphs as a weak proxy
        if not contexts["background"]:
            contexts["background"] = paragraphs[:max(3, len(paragraphs)//10)]
        if not contexts["problem_formulation"]:
            contexts["problem_formulation"] = paragraphs[:max(3, len(paragraphs)//8)]
        if not contexts["methodology"]:
            mid_start = max(0, len(paragraphs)//3)
            contexts["methodology"] = paragraphs[mid_start: mid_start + max(3, len(paragraphs)//8)]
        if not contexts["results_and_findings"]:
            mid_start = max(0, len(paragraphs)//2)
            contexts["results_and_findings"] = paragraphs[mid_start: mid_start + max(3, len(paragraphs)//8)]
        if not contexts["discussion_and_conclusion"]:
            contexts["discussion_and_conclusion"] = paragraphs[-max(3, len(paragraphs)//8):]

        max_chars = int(self.context.get('evidence_max_chars', 20000))
        joined: Dict[str, str] = {}
        for k, paras in contexts.items():
            text_block = "\n\n".join(paras)
            if len(text_block) > max_chars:
                text_block = text_block[:max_chars]
            joined[k] = text_block
        return joined

    async def _extract_evidence_map(self, paper_text: str, agent_name: str = "research_narrative") -> Dict[str, List[str]]:
        contexts = self._build_section_contexts(paper_text)
        prompt = RESEARCH_NARRATIVE_EVIDENCE_PROMPT
        prompt = prompt.replace("{background_text}", contexts.get("background", ""))
        prompt = prompt.replace("{problem_text}", contexts.get("problem_formulation", ""))
        prompt = prompt.replace("{method_text}", contexts.get("methodology", ""))
        prompt = prompt.replace("{results_text}", contexts.get("results_and_findings", ""))
        prompt = prompt.replace("{discussion_text}", contexts.get("discussion_and_conclusion", ""))

        agent_cfg = self.config.agent_configs.get(agent_name, {})
        provider = agent_cfg.get('provider', self.config.default_provider)
        model = agent_cfg.get('model')
        max_tokens = agent_cfg.get('max_tokens', 4000)
        temperature = 0.1
        client = build_client(self.config, provider)
        try:
            response = await client.call(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                retries=self.workflow.get('retry_attempts', 3),
                retry_delay=self.workflow.get('retry_delay', 2),
            )
            evidence = self._parse_json(response)
        except Exception as exc:
            logger.warning(f"Evidence map extraction failed: {exc}")
            evidence = {}
        if not isinstance(evidence, dict):
            evidence = {}

        # Ensure keys exist and are lists
        normalized: Dict[str, List[str]] = {}
        for key in ["background", "problem_formulation", "methodology", "results_and_findings", "discussion_and_conclusion"]:
            items = evidence.get(key, [])
            if isinstance(items, list):
                normalized[key] = [str(i).strip() for i in items if isinstance(i, (str, int, float)) and str(i).strip()]
            else:
                normalized[key] = []
        return normalized

    async def _synthesize_research_narrative(
        self,
        paper_text: str,
        evidence_map: Dict[str, List[str]],
        agent_name: str = "research_narrative",
    ) -> Dict[str, Any]:
        evidence_text = json.dumps(evidence_map, ensure_ascii=False, indent=2)
        prompt = RESEARCH_NARRATIVE_SYNTH_PROMPT.replace("{text}", paper_text).replace("{evidence_map}", evidence_text)
        return await self._call_agent(agent_name, prompt, "", strict_json=True)

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

    async def _extract_research_narrative(self, paper_text: str) -> Dict[str, Any]:
        evidence_map: Dict[str, List[str]] = {}
        primary_ok = True
        try:
            evidence_map = await self._extract_evidence_map(paper_text, "research_narrative")
            narrative = await self._synthesize_research_narrative(paper_text, evidence_map, "research_narrative")
        except Exception as exc:
            logger.warning(f"Research narrative extraction failed: {exc}")
            primary_ok = False
            try:
                evidence_map = await self._extract_evidence_map(paper_text, "research_narrative_fallback")
                narrative = await self._synthesize_research_narrative(paper_text, evidence_map, "research_narrative_fallback")
                logger.warning("Research narrative fallback used: deepseek")
            except Exception as fallback_exc:
                logger.warning(f"Research narrative fallback failed: {fallback_exc}")
                narrative = {}

        if isinstance(narrative, dict):
            narrative = self._ensure_research_narrative_sections(narrative)
            if self._needs_narrative_retry(narrative):
                try:
                    retry_agent = "research_narrative" if primary_ok else "research_narrative_fallback"
                    retry_narrative = await self._synthesize_research_narrative(paper_text, evidence_map, retry_agent)
                    if isinstance(retry_narrative, dict):
                        narrative = self._merge_preserve(narrative, retry_narrative)
                except Exception as exc:
                    logger.warning(f"Research narrative retry failed: {exc}")
            return self._ensure_research_narrative_sections(narrative)
        return {}

    @staticmethod
    def _needs_narrative_retry(narrative: Dict[str, Any]) -> bool:
        fields = [
            ("background", "state_of_the_art_summary"),
            ("problem_formulation", "research_gap"),
            ("problem_formulation", "research_objectives"),
            ("methodology", "method_description"),
            ("results_and_findings", "key_findings"),
            ("discussion_and_conclusion", "final_conclusion"),
        ]
        for section, key in fields:
            node = narrative.get(section, {}).get(key)
            if not isinstance(node, dict):
                return True
            value = node.get("value")
            if value in (None, "", [], {}):
                return True
        return False

    @staticmethod
    def _ensure_research_narrative_sections(narrative: Dict[str, Any]) -> Dict[str, Any]:
        required_sections = {
            "background",
            "problem_formulation",
            "methodology",
            "results_and_findings",
            "discussion_and_conclusion",
        }
        for key in required_sections:
            if key not in narrative or not isinstance(narrative.get(key), dict):
                narrative[key] = {}
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
    def _normalize_supporting_evidence(logic_chain: Dict[str, Any]) -> None:
        rn = logic_chain.get("research_narrative")
        if not isinstance(rn, dict):
            return
        key_findings = rn.get("results_and_findings", {}).get("key_findings")
        if not isinstance(key_findings, dict):
            return
        evidence = key_findings.get("supporting_evidence")
        if not isinstance(evidence, list):
            return
        normalized = []
        for item in evidence:
            if isinstance(item, dict):
                value = item.get("value")
                if isinstance(value, str) and value.strip():
                    normalized.append({"value": value, "source_excerpt": item.get("source_excerpt", "")})
            elif isinstance(item, (str, int, float)):
                text = str(item).strip()
                if text:
                    normalized.append({"value": text, "source_excerpt": ""})
        key_findings["supporting_evidence"] = normalized

    @staticmethod
    def _has_any_keys(obj: Any, keys: set[str]) -> bool:
        if not isinstance(obj, dict):
            return False
        return any(k in obj for k in keys)

    @staticmethod
    def _has_all_keys(obj: Any, keys: set[str]) -> bool:
        if not isinstance(obj, dict):
            return False
        return all(k in obj for k in keys)

    async def extract_file(self, file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        file_text = Path(file_path).read_text(encoding=self.config.get('file_processing', {}).get('text_encoding', 'utf-8'))
        paper_text = self._truncate_text(file_text)
        logger.info("Extraction mode: %s | text_chars=%s", self.extraction_mode, len(paper_text))
        stage_times: Dict[str, float] = {}
        extract_start = time.perf_counter()

        def _is_empty(value: Any) -> bool:
            return value in (None, "", [], {})

        def _empty_dict(obj: Any) -> bool:
            if not isinstance(obj, dict):
                return True
            return all(_is_empty(v) for v in obj.values())

        logic_chain: Dict[str, Any] = {}

        if self.extraction_mode == 'narrative_only':
            t = time.perf_counter()
            metadata = await self._extract_metadata(paper_text)
            stage_times['metadata'] = time.perf_counter() - t

            t = time.perf_counter()
            narrative = await self._extract_research_narrative(paper_text)
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

        elif self.extraction_mode == 'unified':
            unified = await self._call_agent('full_extractor', FULL_EXTRACT_PROMPT_BASE, paper_text, strict_json=True)
            logic_chain = unified if isinstance(unified, dict) else {}
            if not isinstance(logic_chain.get('paper_metadata'), dict):
                logic_chain['paper_metadata'] = {}
            if not isinstance(logic_chain.get('research_narrative'), dict):
                logic_chain['research_narrative'] = {}
            if not isinstance(logic_chain.get('multimedia_content'), dict):
                logic_chain['multimedia_content'] = {}

            if self.enable_targeted_fallback:
                if _empty_dict(logic_chain.get('paper_metadata')):
                    try:
                        metadata = await self._call_agent('metadata', METADATA_PROMPT_BASE, paper_text, strict_json=True)
                        if isinstance(metadata, dict):
                            logic_chain['paper_metadata'] = self._merge_preserve(logic_chain['paper_metadata'], metadata)
                    except Exception as exc:
                        logger.warning(f"Metadata fallback failed: {exc}")

                rn = logic_chain.get('research_narrative', {})
                if _empty_dict(rn.get('background')) or _empty_dict(rn.get('problem_formulation')):
                    try:
                        background = await self._call_agent('background', BACKGROUND_PROMPT_BASE, paper_text, strict_json=True)
                        if isinstance(background, dict):
                            rn = self._merge_preserve(rn, background)
                    except Exception as exc:
                        logger.warning(f"Background fallback failed: {exc}")

                if _empty_dict(rn.get('methodology')):
                    try:
                        methodology = await self._call_agent('methodology', METHODOLOGY_PROMPT_BASE, paper_text, strict_json=True)
                        if isinstance(methodology, dict):
                            rn = self._merge_preserve(rn, methodology)
                    except Exception as exc:
                        logger.warning(f"Methodology fallback failed: {exc}")

                results_block = rn.get('results_and_findings')
                discussion_block = rn.get('discussion_and_conclusion') or rn.get('conclusion')
                if _empty_dict(results_block) or _empty_dict(discussion_block):
                    try:
                        results_and_conclusion = await self._call_agent('results', RESULTS_PROMPT_BASE, paper_text, strict_json=True)
                        if isinstance(results_and_conclusion, dict):
                            rn = self._merge_preserve(rn, results_and_conclusion)
                    except Exception as exc:
                        logger.warning(f"Results fallback failed: {exc}")

                logic_chain['research_narrative'] = rn

                multimedia = logic_chain.get('multimedia_content', {})
                if not self._has_all_keys(multimedia, {"images", "references", "formulas"}):
                    try:
                        multimedia = await self._call_agent('multimedia_content', MULTIMEDIA_CONTENT_PROMPT_BASE, paper_text, strict_json=True)
                    except Exception as exc:
                        logger.warning(f"Multimedia fallback failed: {exc}")
                logic_chain['multimedia_content'] = multimedia

            stage_times['extract'] = time.perf_counter() - extract_start
        else:
            tasks = [
                self._call_agent('metadata', METADATA_PROMPT_BASE, paper_text),
                self._call_agent('background', BACKGROUND_PROMPT_BASE, paper_text),
                self._call_agent('methodology', METHODOLOGY_PROMPT_BASE, paper_text),
                self._call_agent('results', RESULTS_PROMPT_BASE, paper_text),
                self._call_agent('multimedia_content', MULTIMEDIA_CONTENT_PROMPT_BASE, paper_text),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            metadata = results[0] if not isinstance(results[0], Exception) else {}
            background = results[1] if not isinstance(results[1], Exception) else {}
            methodology = results[2] if not isinstance(results[2], Exception) else {}
            results_and_conclusion = results[3] if not isinstance(results[3], Exception) else {}
            multimedia = results[4] if not isinstance(results[4], Exception) else {}

            if not self._has_any_keys(results_and_conclusion, {"results_and_findings", "discussion_and_conclusion", "conclusion"}):
                logger.warning("Results extraction missing, retrying with strict JSON")
                try:
                    results_and_conclusion = await self._call_agent('results', RESULTS_PROMPT_BASE, paper_text, strict_json=True)
                except Exception as exc:
                    logger.warning(f"Results retry failed: {exc}")

            if not self._has_all_keys(multimedia, {"images", "references", "formulas"}):
                logger.warning("Multimedia extraction missing, retrying with strict JSON")
                try:
                    multimedia = await self._call_agent('multimedia_content', MULTIMEDIA_CONTENT_PROMPT_BASE, paper_text, strict_json=True)
                except Exception as exc:
                    logger.warning(f"Multimedia retry failed: {exc}")

            discussion_and_conclusion = results_and_conclusion.get('discussion_and_conclusion', {})
            if not discussion_and_conclusion:
                discussion_and_conclusion = results_and_conclusion.get('conclusion', {})

            logic_chain = {
                'paper_metadata': metadata,
                'research_narrative': {
                    'background': background.get('background', {}),
                    'problem_formulation': background.get('problem_formulation', {}),
                    'methodology': methodology.get('methodology', {}),
                    'results_and_findings': results_and_conclusion.get('results_and_findings', {}),
                    'discussion_and_conclusion': discussion_and_conclusion,
                },
                'multimedia_content': multimedia
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

        if self.extraction_mode != 'narrative_only':
            # Crossref enrichment (metadata only)
            t = time.perf_counter()
            if self.crossref_cfg.get('enable', True):
                logic_chain['paper_metadata'] = await self._crossref_enrich(logic_chain['paper_metadata'])
            stage_times['metadata_enrich'] = time.perf_counter() - t

            if self.keywords_cfg.get('enable', True):
                t = time.perf_counter()
                metadata = logic_chain.get('paper_metadata', {})
                if isinstance(metadata, dict):
                    keywords = metadata.get('keywords')
                    if not keywords:
                        max_keywords = int(self.keywords_cfg.get('max_keywords', 12))
                        language = self.keywords_cfg.get('language', 'en')
                        extracted = extract_keywords(paper_text, max_keywords=max_keywords, language=language)
                        if not extracted and self.keywords_cfg.get('use_llm', True):
                            try:
                                extracted = await self._extract_keywords_with_llm(paper_text, max_keywords)
                            except Exception as exc:
                                logger.warning(f"Keyword LLM fallback failed: {exc}")
                                extracted = []
                        if extracted:
                            metadata['keywords'] = extracted
                            source = metadata.get('metadata_source', {})
                            if isinstance(source, dict):
                                source['llm_supplemented'] = self.keywords_cfg.get('use_llm', True)
                                metadata['metadata_source'] = source
                            logic_chain['paper_metadata'] = metadata
                stage_times['keywords'] = time.perf_counter() - t

        iterative_enabled = bool(self.iterative_cfg.get('enable', False))
        max_rounds = int(self.iterative_cfg.get('max_rounds', 3))
        stop_on_no_change = bool(self.iterative_cfg.get('stop_on_no_change', True))

        if iterative_enabled and self.extraction_mode == 'narrative_only':
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

        elif iterative_enabled:
            t = time.perf_counter()
            last_hash = None
            threshold = int(self.quality_cfg.get('threshold', 75))
            for round_idx in range(max_rounds):
                errors = self.validator.validate(logic_chain)
                needs_repair = bool(errors) and self.config.get('repair', {}).get('enable_schema_repair', True)
                needs_quality = self.quality_cfg.get('enable', True)
                final_score = None
                if needs_quality:
                    final_score = await self._compute_quality_score(logic_chain, paper_text)
                needs_refine = needs_repair or (final_score is not None and final_score < threshold)
                logger.info(
                    "Iterative refine round %s/%s: errors=%s score=%s",
                    round_idx + 1,
                    max_rounds,
                    len(errors),
                    final_score if final_score is not None else "n/a",
                )
                if not needs_refine:
                    break

                before = json.dumps(logic_chain, ensure_ascii=False, sort_keys=True)
                if needs_repair:
                    logic_chain = await self._repair_json_with_llm(logic_chain, paper_text)
                if final_score is not None and final_score < threshold:
                    logic_chain = await self._refine_with_llm(logic_chain, paper_text)
                after = json.dumps(logic_chain, ensure_ascii=False, sort_keys=True)
                if stop_on_no_change and after == before:
                    logger.info("Iterative refine stopped: no changes in round %s", round_idx + 1)
                    break
                last_hash = after
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

        # Normalize segment_map structure for merged fields
        self._normalize_segment_map_fields(logic_chain)
        self._normalize_metadata_lists(logic_chain)
        self._normalize_supporting_evidence(logic_chain)

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
                        OPTIONAL MATCH (p)-[:HAS_RESEARCH_NARRATIVE]->(rn)
                        OPTIONAL MATCH (p)-[:HAS_MULTIMEDIA_CONTENT]->(mm)
                        OPTIONAL MATCH (p)-[:CITES]->(c)
                        RETURN count(rn) as rn, count(mm) as mm, count(c) as cites
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
                        "has_research_narrative": bool(row.get("rn")),
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
                    if integrity.get("found") and (not integrity.get("has_research_narrative") or not integrity.get("has_multimedia_content")):
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
