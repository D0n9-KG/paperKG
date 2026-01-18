"""JSON Schema validation utilities."""
from __future__ import annotations

from typing import Any, Dict, List

from jsonschema import Draft202012Validator


class SchemaValidator:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.validator = Draft202012Validator(schema)

    def validate(self, data: Dict[str, Any]) -> List[str]:
        errors = []
        for err in sorted(self.validator.iter_errors(data), key=lambda e: e.path):
            path = '.'.join([str(p) for p in err.path])
            errors.append(f"{path}: {err.message}")
        return errors
