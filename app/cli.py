"""CLI entrypoint for PaperKG."""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from core.config import Config
from pipeline.orchestrator import PaperKGExtractor


def setup_logging(config: Config) -> None:
    cfg = config.get('logging', {})
    level_name = str(cfg.get('level', 'INFO')).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = cfg.get('console_format', '%(levelname)s: %(message)s')
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger('core.crossref').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('neo4j').setLevel(logging.WARNING)


def _collect_files(source_path: Path, supported_exts: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in source_path.rglob('*'):
        if p.suffix.lower() in supported_exts:
            files.append(p)
    return sorted(files)


async def _process_file(
    extractor: PaperKGExtractor,
    path: Path,
    output_dir: Optional[Path],
    idx: int,
    total: int,
    pbar: Optional[tqdm],
    semaphore: asyncio.Semaphore,
) -> None:
    logger = logging.getLogger(__name__)
    start = time.time()
    label = f"[{idx}/{total}] {path.name}"
    if pbar:
        tqdm.write(f"START {label}")
    else:
        logger.info(f"START {label}")

    async with semaphore:
        try:
            out_path = None
            if output_dir:
                output_dir.mkdir(exist_ok=True)
                out_path = output_dir / f"{path.stem}_logic_chain.json"
            await extractor.extract_file(str(path), str(out_path) if out_path else None)
            status = "OK"
        except Exception as exc:
            status = f"FAIL: {exc}"
            logger.warning(f"{label} failed: {exc}")

    elapsed = time.time() - start
    if pbar:
        tqdm.write(f"DONE  {label}  ({elapsed:.1f}s)  {status}")
        pbar.update(1)
    else:
        logger.info(f"DONE  {label}  ({elapsed:.1f}s)  {status}")


async def _process_directory(
    extractor: PaperKGExtractor,
    files: List[Path],
    output_dir: Optional[Path],
    parallel_limit: int,
    enable_progress: bool,
) -> None:
    logger = logging.getLogger(__name__)
    total = len(files)
    if total == 0:
        logger.warning("No supported files found.")
        return

    logger.info(f"Discovered {total} files. Parallel limit: {parallel_limit}")
    pbar = tqdm(total=total, desc="Extracting", unit="file") if enable_progress else None
    semaphore = asyncio.Semaphore(parallel_limit)

    tasks = [
        _process_file(extractor, path, output_dir, idx + 1, total, pbar, semaphore)
        for idx, path in enumerate(files)
    ]
    await asyncio.gather(*tasks)
    if pbar:
        pbar.close()
    logger.info("Batch extraction finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description='PaperKG - Paper Logic Chain Extractor')
    parser.add_argument('source', help='Source file or directory')
    parser.add_argument('--output-dir', '-o', help='Output directory')
    parser.add_argument('--config', default='config/default.yaml', help='Config path')
    parser.add_argument('--schema', default='config/output_schema.json', help='Schema path')
    args = parser.parse_args()

    config = Config(args.config)
    setup_logging(config)

    extractor = PaperKGExtractor(config_path=args.config, schema_path=args.schema)
    source_path = Path(args.source)
    output_dir = Path(args.output_dir) if args.output_dir else None

    supported_exts = config.get('source_files', {}).get('supported_extensions', ['.md', '.markdown', '.txt'])
    supported_exts = [ext.lower() for ext in supported_exts]

    if source_path.is_dir():
        files = _collect_files(source_path, supported_exts)
        parallel_limit = int(config.get('workflow', {}).get('batch_parallel_limit', 3))
        enable_progress = bool(config.get('logging', {}).get('enable_progress_bar', True))
        asyncio.run(_process_directory(extractor, files, output_dir, parallel_limit, enable_progress))
    else:
        out_path = None
        if output_dir:
            output_dir.mkdir(exist_ok=True)
            out_path = output_dir / f"{source_path.stem}_logic_chain.json"
        extractor.extract_file_sync(str(source_path), str(out_path) if out_path else None)


if __name__ == '__main__':
    main()
