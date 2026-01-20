"""Prompt builder that injects output format from JSON Schema."""
from __future__ import annotations

from typing import Dict, List

from core.schema import SchemaLoader, OutputFormatBuilder


AGENT_SCHEMA_PATHS = {
    'metadata': ['paper_metadata'],
    'background': ['research_narrative'],
    'methodology': ['research_narrative'],
    'results': ['research_narrative'],
    'multimedia_content': ['multimedia_content'],
    'full_extractor': [],
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

        # Narrow to agent-specific fields if needed
        if agent_name in {'background', 'methodology', 'results'} and subschema.get('type') == 'object':
            props = subschema.get('properties', {})
            if agent_name == 'background':
                subschema = {
                    'type': 'object',
                    'properties': {
                        'background': props.get('background', {'type': 'object'}),
                        'problem_formulation': props.get('problem_formulation', {'type': 'object'})
                    },
                    'required': ['background', 'problem_formulation'],
                    'additionalProperties': False
                }
            elif agent_name == 'methodology':
                subschema = {
                    'type': 'object',
                    'properties': {
                        'methodology': props.get('methodology', {'type': 'object'})
                    },
                    'required': ['methodology'],
                    'additionalProperties': False
                }
            elif agent_name == 'results':
                subschema = {
                    'type': 'object',
                    'properties': {
                        'results_and_findings': props.get('results_and_findings', {'type': 'object'}),
                        'discussion_and_conclusion': props.get('discussion_and_conclusion', {'type': 'object'})
                    },
                    'required': ['results_and_findings', 'discussion_and_conclusion'],
                    'additionalProperties': False
                }
        return subschema

    def build(self, agent_name: str, base_prompt: str) -> str:
        subschema = self.get_schema(agent_name)
        if not subschema:
            return base_prompt.replace('{OUTPUT_FORMAT_SECTION}', '')
        # Narrow to agent-specific fields if needed
        if agent_name in {'background', 'methodology', 'results'} and subschema.get('type') == 'object':
            props = subschema.get('properties', {})
            if agent_name == 'background':
                subschema = {
                    'type': 'object',
                    'properties': {
                        'background': props.get('background', {'type': 'object'}),
                        'problem_formulation': props.get('problem_formulation', {'type': 'object'})
                    },
                    'required': ['background', 'problem_formulation'],
                    'additionalProperties': False
                }
            elif agent_name == 'methodology':
                subschema = {
                    'type': 'object',
                    'properties': {
                        'methodology': props.get('methodology', {'type': 'object'})
                    },
                    'required': ['methodology'],
                    'additionalProperties': False
                }
            elif agent_name == 'results':
                subschema = {
                    'type': 'object',
                    'properties': {
                        'results_and_findings': props.get('results_and_findings', {'type': 'object'}),
                        'discussion_and_conclusion': props.get('discussion_and_conclusion', {'type': 'object'})
                    },
                    'required': ['results_and_findings', 'discussion_and_conclusion'],
                    'additionalProperties': False
                }

        example = OutputFormatBuilder(subschema).build()
        section = "\n\n【输出格式】\n请严格按以下JSON模板输出：\n```json\n" + example + "\n```\n\n只输出JSON。"
        return base_prompt.replace('{OUTPUT_FORMAT_SECTION}', section)
