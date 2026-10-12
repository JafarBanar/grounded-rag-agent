"""Citation checking: every claim must cite a provision the agent actually retrieved.

Answers cite unit ids in square brackets, e.g. ``[AI Act Art. 6(2)]``. The check is
done by `groundcite` with a pattern for that syntax; this module decides which ids
count as retrieved (the evidence ledger) and adds a separate check for ids that do
not exist in the corpus at all (invented citations).
"""

from __future__ import annotations

import re

from groundcite import validate
from groundcite.patterns import CitationPattern

from .index import HybridIndex


def canon(ref: str) -> str:
    """Canonical form for comparing ids: lower case, single spaces, 'Article N' -> 'art. n'."""
    r = ref.replace(PROTECT, ". ")
    r = " ".join(r.replace("\xa0", " ").split()).lower()
    r = re.sub(r"\barticle\s+", "art. ", r)
    r = re.sub(r"\bart\s+(?=\d)", "art. ", r)
    return r


# groundcite splits sentences at ". " which would cut "[GDPR Art. 22(1)]" in two.
# Inside brackets, ". " is replaced by this marker before checking, and restored by canon().
PROTECT = ".\u2060"  # period + word joiner (not whitespace)


def protect_citations(text: str) -> str:
    return re.sub(r"\[[^\[\]\n]{2,140}\]", lambda m: m.group(0).replace(". ", PROTECT), text)


# [id] but not a Markdown link [text](url); ids never contain brackets themselves.
BRACKET = CitationPattern(
    name="bracket-id",
    regex=re.compile(r"\[(?P<ref>[^\[\]\n]{2,140})\](?!\()"),
    normalise=canon,
)


def allowed_ids(index: HybridIndex, retrieved: set[str]) -> set[str]:
    """Ids the answer may cite: retrieved units, their ancestors, and children of retrieved containers."""
    allowed: set[str] = set()
    for uid in retrieved:
        allowed.add(uid)
        allowed.update(index.ancestors(uid))
        stack = list(index.children.get(uid, []))
        while stack:
            c = stack.pop()
            allowed.add(c)
            stack.extend(index.children.get(c, []))
    return allowed


def check(answer: str, index: HybridIndex, retrieved: set[str], min_coverage: float = 0.8) -> dict:
    """Return a report dict: ok flag, problems, cited ids, invented ids, coverage."""
    allowed = allowed_ids(index, retrieved)
    report = validate(protect_citations(answer), [canon(a) for a in allowed], pattern=BRACKET, min_coverage=min_coverage)
    known = {canon(u.id) for u in index.units}
    invented = [r for r in report.cited_refs if r not in known]
    problems = list(report.problems)
    if invented:
        problems.append("citations that do not exist in the corpus: " + ", ".join(invented))
    by_canon = {canon(u.id): u.id for u in index.units}
    return {
        "ok": report.ok and not invented,
        "problems": problems,
        "cited": [by_canon.get(r, r) for r in report.cited_refs],
        "unresolved": [by_canon.get(r, r) for r in report.unresolved_refs],
        "invented": invented,
        "coverage": round(report.coverage, 3),
        "uncited_sentences": [x.replace(PROTECT, ". ") for x in report.uncited_sentences],
    }
