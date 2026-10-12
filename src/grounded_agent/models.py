"""Core data types shared by ingestion, indexing, tools and the agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Unit:
    """The smallest citable piece of a document.

    ``id`` is also the citation string the model writes in square brackets,
    for example ``AI Act Art. 6(2)`` or ``GDPR Art. 9(1)``.
    """

    id: str
    doc: str                      # document label, e.g. "AI Act"
    kind: str                     # article | paragraph | point | recital | annex | section | page
    title: str                    # human title, e.g. "Classification rules for high-risk AI systems"
    text: str
    parent: str | None = None     # article-level id for paragraphs and points
    refs: list[str] = field(default_factory=list)   # ids this unit refers to (resolved later)
    source: str = ""              # file or URL the text came from

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Unit":
        return cls(**data)


@dataclass
class Hit:
    """One search result."""

    unit: Unit
    score: float
    rank: int
