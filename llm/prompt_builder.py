"""Prompt builder that injects output format from JSON Schema."""
from __future__ import annotations

from typing import Dict, List

from core.schema import SchemaLoader, OutputFormatBuilder


AGENT_SCHEMA_PATHS = {
    'metadata': ['paper_metadata'],
    'multimedia_content': ['multimedia_content'],
    'research_narrative': ['research_narrative'],
    'research_narrative_fallback': ['research_narrative']
}


class PromptBuilder:
    def __init__(self, schema_loader: SchemaLoader):
        self.schema_loader = schema_loader

    def get_schema(self, agent_name: str) -> Dict:
        if agent_name.endswith("_selector"):
            return {}
        path = AGENT_SCHEMA_PATHS.get(agent_name, ['$root'])
        subschema = self.schema_loader.get_subschema(['$root'] + path)
        if not subschema:
            return {}

        return subschema

    def build(self, agent_name: str, base_prompt: str) -> str:
        subschema = self.get_schema(agent_name)
        if not subschema:
            return base_prompt.replace('{OUTPUT_FORMAT_SECTION}', '')
        example = OutputFormatBuilder(subschema).build()
        section = "\n\n【输出格式】\n请严格按以下JSON模板输出：\n```json\n" + example + "\n```\n\n只输出JSON。"
        return base_prompt.replace('{OUTPUT_FORMAT_SECTION}', section)
