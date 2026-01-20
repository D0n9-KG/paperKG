import json
from pathlib import Path
import unittest

from pipeline.orchestrator import PaperKGExtractor
from storage.neo4j.mapper import GraphMapper


class FakeCrossrefClient:
    async def get_metadata(self, doi=None, title=None, authors=None, year=None, full_citation=None):
        if full_citation:
            return {
                "DOI": "10.9999/example.2020.001",
                "URL": "https://doi.org/10.9999/example.2020.001",
                "title": ["Example Widgets Study"],
                "author": [{"given": "Alice", "family": "Zhang"}],
                "published": {"date-parts": [[2020, 1, 1]]}
            }
        # Minimal Crossref payload for extractor
        return {
            "DOI": doi or "10.1234/widget.2024.001",
            "URL": "https://doi.org/10.1234/widget.2024.001",
            "title": [title or "Test Paper on Neural Widgets"],
            "author": [{"given": "Alice", "family": "Zhang"}],
            "published": {"date-parts": [[2024, 1, 1]]}
        }


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).parent
        self.sample_file = self.base_dir / "fixtures" / "sample.txt"
        self.sample_output = json.loads((self.base_dir / "fixtures" / "sample_output.json").read_text(encoding='utf-8'))
        self.config_path = str(self.base_dir / "fixtures" / "test_config.yaml")
        self.schema_path = "config/output_schema.json"

    def test_end_to_end_pipeline(self):
        extractor = PaperKGExtractor(config_path=self.config_path, schema_path=self.schema_path)

        # Patch LLM calls
        async def fake_call_agent(agent_name, base_prompt, text, strict_json=False):
            if agent_name == 'metadata':
                return self.sample_output['paper_metadata']
            if agent_name == 'research_narrative':
                return self.sample_output['research_narrative']
            if agent_name == 'multimedia_content':
                return self.sample_output['multimedia_content']
            return {}

        extractor._call_agent = fake_call_agent
        extractor.crossref_client = FakeCrossrefClient()

        output_dir = self.sample_file.parent / "output"
        output_path = output_dir / f"{self.sample_file.stem}_logic_chain.json"
        sidecar_path = output_dir / f"{self.sample_file.stem}_logic_chain_evidence_segments.json"
        coverage_path = output_dir / f"{self.sample_file.stem}_logic_chain_coverage.json"
        if output_path.exists():
            output_path.unlink()
        if sidecar_path.exists():
            sidecar_path.unlink()
        if coverage_path.exists():
            coverage_path.unlink()

        try:
            result = extractor.extract_file_sync(str(self.sample_file))
            self.assertIn('paper_metadata', result)
            self.assertIn('research_narrative', result)
            self.assertIn('multimedia_content', result)
            self.assertTrue(output_path.exists())

            refs = result.get("multimedia_content", {}).get("references", {}).get("reference_list", [])
            self.assertTrue(refs)
            self.assertEqual(refs[0].get("doi"), "10.9999/example.2020.001")
        finally:
            if output_path.exists():
                output_path.unlink()
            if sidecar_path.exists():
                sidecar_path.unlink()
            if coverage_path.exists():
                coverage_path.unlink()

    def test_mapper_covers_all_fields(self):
        mapper = GraphMapper(conn=None)
        result = mapper.map(self.sample_output, dry_run=True)
        visited = set(result['visited_paths'])

        # Collect scalar paths from sample output
        def collect_paths(obj, prefix=""):
            paths = set()
            if isinstance(obj, dict):
                for k, v in obj.items():
                    p = f"{prefix}.{k}" if prefix else k
                    paths |= collect_paths(v, p)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    p = f"{prefix}[{i}]"
                    paths |= collect_paths(v, p)
            else:
                paths.add(prefix)
            return paths

        all_paths = collect_paths(self.sample_output)
        self.assertEqual(all_paths, visited)


if __name__ == '__main__':
    unittest.main()
