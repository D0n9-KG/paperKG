"""Main extraction pipeline."""
from __future__ import annotations

import asyncio
import json
import logging
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
            "- 不要臆造，不要使用空泛表述\n"
            "- 允许更详细，但必须忠实原文\n"
        )

        if strict_json:
            prompt += (
                "\n\n[STRICT JSON OUTPUT]\n"
                "- Output JSON only. No explanations, titles, comments, or Markdown.\n"
                "- Do not include non-JSON characters or trailing text.\n"
            )

        response = await client.call(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=self.workflow.get('retry_attempts', 3),
            retry_delay=self.workflow.get('retry_delay', 2),
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
        for idx, ch in enumerate(text):
            if ch in ".!????":
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
        tokens = group.replace(";", ",").split(",")
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if token == ref_id:
                return True
            # handle ranges like 4-9 or 4~9
            for sep in ("-", "~"):
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
        # bracketed numeric citations
        import re
        for match in re.finditer(r"\[(.*?)\]", text):
            group = match.group(1)
            if self._ref_id_in_group(ref_id, group):
                return self._context_from_span(text, match.start(), match.end(), window)
        # fallback: try surname from citation
        if citation_text:
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

        # Enrich reference DOIs via Crossref when missing
        multimedia = await self._enrich_reference_dois(multimedia)
        # Add LLM-based citation purpose summaries
        multimedia = await self._extract_citation_purposes(paper_text, multimedia)

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

        # Crossref enrichment (metadata only)
        if self.crossref_cfg.get('enable', True):
            logic_chain['paper_metadata'] = await self._crossref_enrich(logic_chain['paper_metadata'])

        if self.keywords_cfg.get('enable', True):
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

        # Quality scoring + refinement
        if self.quality_cfg.get('enable', True):
            rule_sc, rule_issues = rule_score(logic_chain)
            llm_sc = None
            if self.quality_cfg.get('use_llm', True):
                llm_res = await self._score_with_llm(logic_chain, paper_text)
                if isinstance(llm_res, dict):
                    llm_sc = llm_res.get('score')
            weight_r = float(self.quality_cfg.get('rule_weight', 0.4))
            weight_l = float(self.quality_cfg.get('llm_weight', 0.6))
            final_score = rule_sc if llm_sc is None else int(rule_sc * weight_r + llm_sc * weight_l)
            threshold = int(self.quality_cfg.get('threshold', 75))
            if final_score < threshold:
                rounds = int(self.quality_cfg.get('refine_rounds', 1))
                for _ in range(max(1, rounds)):
                    logic_chain = await self._refine_with_llm(logic_chain, paper_text)

        # Optional schema repair
        if self.config.get('repair', {}).get('enable_schema_repair', True):
            logic_chain = await self._repair_json_with_llm(logic_chain, paper_text)

        # Validate
        errors = self.validator.validate(logic_chain)
        if errors:
            logger.warning("Schema validation errors:\n" + "\n".join(errors[:20]))

        # Write output
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

        if self.neo4j_cfg.get('enable', False):
            try:
                from storage.exporters.neo4j_exporter import convert_paper_to_neo4j
                convert_paper_to_neo4j(logic_chain, self.neo4j_cfg.get('mapping_path', 'config/neo4j_mapping.json'))
            except Exception as exc:
                logger.warning(f"Neo4j export failed: {exc}")
        return logic_chain

    def extract_file_sync(self, file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        if self.workflow.get('enable_async', True):
            return asyncio.run(self.extract_file(file_path, output_path))
        return asyncio.run(self.extract_file(file_path, output_path))
