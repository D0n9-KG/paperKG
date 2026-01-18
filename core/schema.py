"""
JSON Schema loader and helper utilities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class SchemaLoader:
    def __init__(self, schema_path: str | Path):
        self.schema_path = Path(schema_path)
        self.schema = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {self.schema_path}")
        return json.loads(self.schema_path.read_text(encoding='utf-8'))

    def get_subschema(self, path: List[str]) -> Dict[str, Any]:
        node = self.schema
        for key in path:
            if key == '$root':
                continue
            if 'properties' in node and key in node['properties']:
                node = node['properties'][key]
            else:
                return {}
        return node


class OutputFormatBuilder:
    """Generate a compact output format section for prompts."""

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def _example_for_schema(self, schema: Dict[str, Any]) -> Any:
        stype = schema.get('type')
        if isinstance(stype, list):
            stype = [t for t in stype if t != 'null']
            stype = stype[0] if stype else 'string'

        if stype == 'object':
            props = schema.get('properties', {})
            example = {}
            for k, v in props.items():
                example[k] = self._example_for_schema(v)
            return example
        if stype == 'array':
            items = schema.get('items', {})
            return [self._example_for_schema(items)]
        if stype == 'string':
            return "<string>"
        if stype == 'integer':
            return 0
        if stype == 'number':
            return 0.0
        if stype == 'boolean':
            return False
        return "<value>"

    def build(self) -> str:
        example = self._example_for_schema(self.schema)
        return json.dumps(example, ensure_ascii=False, indent=2)
