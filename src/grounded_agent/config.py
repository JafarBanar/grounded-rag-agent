"""Corpus and runtime settings, loaded from YAML files in ``corpora/``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Source:
    path: str
    doc: str
    kind: str = "files"        # "eurlex" for EU acts, "files" for private documents


@dataclass
class CorpusConfig:
    name: str
    description: str           # tells the model what the documents are
    citation_example: str      # e.g. "[AI Act Art. 6(2)]"
    sources: list[Source]
    index_dir: Path
    model: str = "gemma4:e4b"
    embed_model: str = "embeddinggemma"
    think: bool = False
    max_steps: int = 8
    max_tool_calls: int = 12
    min_coverage: float = 0.8
    repair_attempts: int = 1
    extra_rules: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "CorpusConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        base = Path(path).resolve().parent.parent
        raw["sources"] = [Source(**s) for s in raw["sources"]]
        for s in raw["sources"]:
            p = Path(s.path).expanduser()
            s.path = str(p if p.is_absolute() else (base / p))
        idx = Path(raw.pop("index_dir", f"data/index/{raw['name']}")).expanduser()
        raw["index_dir"] = idx if idx.is_absolute() else base / idx
        return cls(**raw)
