"""Build (or rebuild) the index for one corpus: parse, resolve references, embed, save."""

from __future__ import annotations

from pathlib import Path

from .config import CorpusConfig
from .index import HybridIndex
from .ingest.eurlex import parse_act
from .ingest.files import load_path
from .ingest.refs import resolve
from .llm import OllamaClient


def load_units(cfg: CorpusConfig):
    units = []
    for s in cfg.sources:
        p = Path(s.path)
        if not p.exists():
            raise FileNotFoundError(f"source not found: {p} (run scripts/download_eu_texts.sh for the EU corpus)")
        units += parse_act(p, s.doc) if s.kind == "eurlex" else load_path(p, s.doc)
    known = {u.id for u in units}
    for u in units:
        u.refs = resolve(u.refs, known, u.id, u.parent)
    return units


def build(cfg: CorpusConfig, embed: bool = True, log=print) -> HybridIndex:
    units = load_units(cfg)
    log(f"{cfg.name}: {len(units)} units from {len(cfg.sources)} sources")
    index = HybridIndex(units)
    if embed:
        client = OllamaClient(cfg.model, cfg.embed_model)
        log(f"embedding {len(index.search_ids)} units with {cfg.embed_model} ...")
        index.embed_all(client.embed)
    index.save(Path(cfg.index_dir))
    log(f"saved index to {cfg.index_dir}")
    return index
