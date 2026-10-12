"""Baseline for comparison: classic retrieve-once RAG with the same index, model and citation check."""

from __future__ import annotations

import time
from pathlib import Path

from .agent import NOT_FOUND, Result, refusal_text
from .citations import check, expand_multi_citations
from .config import CorpusConfig
from .index import HybridIndex
from .llm import OllamaClient

PROMPT = """Answer the question using only the passages below. The documents are: {description}

End every factual sentence with the id of its passage in square brackets, exactly as shown, for example {citation_example}.
If the passages do not answer the question, reply with "{not_found}" and one sentence.
Be concise.

Passages:
{passages}

Question: {question}"""


class Workflow:
    def __init__(self, cfg: CorpusConfig, index: HybridIndex | None = None, client: OllamaClient | None = None, k: int = 8):
        self.cfg, self.k = cfg, k
        self.index = index or HybridIndex.load(Path(cfg.index_dir))
        self.client = client or OllamaClient(cfg.model, cfg.embed_model, think=cfg.think)

    def run(self, question: str, on_event=None) -> Result:
        t0 = time.perf_counter()
        hits = self.index.search(question, k=self.k, embed_fn=self.client.embed)
        retrieved = {h.unit.id for h in hits}
        passages = "\n\n".join(f"[{h.unit.id}] {h.unit.title}\n{h.unit.text[:1500]}" for h in hits)
        prompt = PROMPT.format(description=self.cfg.description.strip(), citation_example=self.cfg.citation_example,
                               not_found=NOT_FOUND, passages=passages, question=question)
        r = self.client.chat([{"role": "user", "content": prompt}])
        answer = r.content.strip()
        res = Result(question=question, answer=answer, status="error", mode="workflow", model_calls=1,
                     prompt_tokens=r.prompt_tokens, output_tokens=r.output_tokens, retrieved=sorted(retrieved), draft=answer)
        if answer.upper().startswith(NOT_FOUND):
            res.status = "not_found"
        else:
            rep = check(answer, self.index, retrieved, self.cfg.min_coverage)
            res.validation = rep
            if rep["ok"]:
                res.status, res.cited = "answered", rep["cited"]
                res.answer = expand_multi_citations(answer, sorted({u.doc for u in self.index.units}))
            else:
                res.status, res.answer = "refused", refusal_text(sorted(retrieved))
        res.seconds = round(time.perf_counter() - t0, 2)
        return res
