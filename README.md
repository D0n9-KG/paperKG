# PaperKG (Rebuilt)

A rebuilt and modular Paper Logic Chain Extractor.

## Environment

```bash
conda activate paperKG
pip install -r requirements.txt
```

## Run

```bash
python -m app.cli path/to/paper.md
# or
python -m app.cli source_dir/ --output-dir output
```

## Configuration
- `config/default.yaml` for providers, workflow, Crossref, Neo4j.
- `config/output_schema.json` for JSON Schema.
- `config/neo4j_mapping.json` for Neo4j mapping.

## Notes
- Crossref enrichment is enabled by default.
- Neo4j export is optional: set `neo4j.enable: true`.
