from pathlib import Path

import pytest

from grounded_agent.config import CorpusConfig, Source
from grounded_agent.index import HybridIndex
from grounded_agent.ingest.eurlex import parse_act
from grounded_agent.ingest.refs import resolve

FIXTURE = Path(__file__).parent / "fixtures" / "mini_act.xhtml"


@pytest.fixture
def units():
    us = parse_act(FIXTURE, "AI Act")
    known = {u.id for u in us}
    for u in us:
        u.refs = resolve(u.refs, known, u.id, u.parent)
    return us


@pytest.fixture
def index(units):
    return HybridIndex(units)


@pytest.fixture
def cfg(tmp_path):
    return CorpusConfig(name="test", description="A test act.", citation_example="[AI Act Art. 6(2)]",
                        sources=[Source(path=str(FIXTURE), doc="AI Act", kind="eurlex")], index_dir=tmp_path,
                        min_coverage=1.0, repair_attempts=1, plan=False)
