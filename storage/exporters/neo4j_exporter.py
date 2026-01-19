"""Neo4j exporter using Paper-centric mapper with evidence nodes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import logging

from storage.neo4j.neo4j_utils import get_neo4j_connection
from storage.neo4j.mapper import GraphMapper


class PaperToNeo4jConverter:
    def __init__(self, mapping_config_path: str = "config/neo4j_mapping.json"):
        self.mapping_config_path = Path(mapping_config_path)
        self.mapping_config = self._load_mapping_config()
        self.conn = get_neo4j_connection()
        self.mapper = GraphMapper(self.conn)

    def _load_mapping_config(self) -> Dict[str, Any]:
        if not self.mapping_config_path.exists():
            return {}
        return json.loads(self.mapping_config_path.read_text(encoding='utf-8-sig'))

    def convert_paper(self, paper_data: Dict[str, Any]) -> bool:
        try:
            self.mapper.map(paper_data, dry_run=False)
            return True
        except Exception as exc:
            logging.getLogger(__name__).warning("Neo4j export failed: %s", exc)
            return False


def convert_paper_to_neo4j(paper_data: Dict[str, Any], mapping_config_path: str = "config/neo4j_mapping.json") -> bool:
    converter = PaperToNeo4jConverter(mapping_config_path)
    return converter.convert_paper(paper_data)


def convert_paper_file_to_neo4j(paper_file_path: str, mapping_config_path: str = "config/neo4j_mapping.json") -> bool:
    data = json.loads(Path(paper_file_path).read_text(encoding='utf-8'))
    return convert_paper_to_neo4j(data, mapping_config_path)
