"""Hybrid retrieval: BM25 for exact legal terms, embeddings for meaning, fused with RRF."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .models import Hit, Unit

_TOKEN = re.compile(r"[a-z0-9]+(?:[\-/][a-z0-9]+)*")
STOP = set("""a an and are as at be by for from has have in is it its of on or that the this to was were will with
shall such which may any other where those these their there be been being not no into under within than who
whose whom also only all each per more most can could should would""".split())


def stem(t: str) -> str:
    """Light, predictable suffix stripping: application/applies/applied -> apply, notifications -> notify."""
    if any(ch.isdigit() for ch in t) or len(t) <= 4:
        return t
    for suf, rep in (("ications", "y"), ("ication", "y"), ("ies", "y"), ("ied", "y"), ("sses", "ss")):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)] + rep
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    for suf in ("ing", "ed"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def tokenize(text: str) -> list[str]:
    """Lower-case, lightly stemmed word tokens; keeps codes such as 2016/679 and high-risk intact, drops stop words."""
    toks = _TOKEN.findall(text.lower())
    out = []
    for t in toks:
        if t in STOP or len(t) == 1:
            continue
        out.append(stem(t))
        if "-" in t:                      # "high-risk" also matches "high risk"
            out.extend(stem(p) for p in t.split("-") if p and p not in STOP)
    return out


class BM25:
    def __init__(self, k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []

    def fit(self, docs: list[list[str]]) -> "BM25":
        for i, toks in enumerate(docs):
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.n = len(docs)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        return self

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for term in set(query):
            idf = self.idf(term)
            for i, tf in self.postings.get(term, ()):
                dl = self.doc_len[i]
                s[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


def _embed_text(u: Unit) -> str:
    return f"{u.id} {u.title}: {u.text}"[:6000]


class HybridIndex:
    """Units plus a BM25 index and (optionally) a dense embedding matrix."""

    def __init__(self, units: list[Unit], vectors: np.ndarray | None = None):
        self.units = units
        self.by_id = {u.id: u for u in units}
        self.children: dict[str, list[str]] = defaultdict(list)
        for u in units:
            if u.parent:
                self.children[u.parent].append(u.id)
        # Search over citable leaves and recitals; whole articles/annexes are for reading.
        self.search_ids = [i for i, u in enumerate(units)
                           if u.kind not in ("article", "annex") or not self.children.get(u.id)]
        self.bm25 = BM25().fit([tokenize(f"{units[i].title} {units[i].text}") for i in self.search_ids])
        if vectors is not None and len(vectors) != len(self.search_ids):
            raise ValueError(f"index vectors ({len(vectors)}) do not match searchable units ({len(self.search_ids)}); "
                             "rebuild with `gra build <corpus>`")
        self.vectors = vectors

    # ----------------------------------------------------------- persistence
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "units.jsonl").write_text("\n".join(json.dumps(u.to_dict(), ensure_ascii=False) for u in self.units), encoding="utf-8")
        if self.vectors is not None:
            np.save(folder / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, folder: Path) -> "HybridIndex":
        units = [Unit.from_dict(json.loads(l)) for l in (folder / "units.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        vec_path = folder / "vectors.npy"
        vectors = np.load(vec_path) if vec_path.exists() else None
        return cls(units, vectors)

    def embed_all(self, embed_fn, prefix: str = "title: none | text: ") -> None:
        texts = [prefix + _embed_text(self.units[i]) for i in self.search_ids]
        vecs = np.asarray(embed_fn(texts), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        self.vectors = vecs

    # ----------------------------------------------------------- search
    def search(self, query: str, k: int = 6, embed_fn=None, doc: str | None = None,
               query_prefix: str = "task: search result | query: ", rrf_k: int = 60, per_method: int = 2) -> list[Hit]:
        """Hybrid search with reciprocal-rank fusion. Falls back to BM25 when no vectors."""
        bm = self.bm25.scores(tokenize(query))
        ranked_lists = [np.argsort(-bm)[:200]]
        if self.vectors is not None and embed_fn is not None:
            q = np.asarray(embed_fn([query_prefix + query])[0], dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-9
            dense = self.vectors @ q
            ranked_lists.append(np.argsort(-dense)[:200])
        fused: dict[int, float] = defaultdict(float)
        for lst in ranked_lists:
            for r, j in enumerate(lst):
                if len(ranked_lists) == 1 and bm[j] <= 0:
                    break
                fused[int(j)] += 1.0 / (rrf_k + r + 1)
        order = sorted(fused, key=lambda j: -fused[j])
        # A strong winner of one method can lose in plain RRF when the other method ranks it low;
        # always keep the top `per_method` of each list, then fill by fused score.
        lead: list[int] = []
        for lst in ranked_lists:
            for j in lst[:per_method]:
                if int(j) in fused and int(j) not in lead:
                    lead.append(int(j))
        order = lead + [j for j in order if j not in lead]
        hits: list[Hit] = []
        shown_parents: set[str] = set()
        for j in order:
            u = self.units[self.search_ids[j]]
            if doc and u.doc.lower() != doc.lower():
                continue
            # avoid showing both a paragraph and its own lettered points
            if u.id in shown_parents or (u.parent and u.parent in {h.unit.id for h in hits}):
                continue
            shown_parents.add(u.parent or "")
            hits.append(Hit(unit=u, score=fused[j], rank=len(hits) + 1))
            if len(hits) >= k:
                break
        return hits

    def ancestors(self, unit_id: str) -> list[str]:
        out, u = [], self.by_id.get(unit_id)
        while u and u.parent:
            out.append(u.parent)
            u = self.by_id.get(u.parent)
        return out
