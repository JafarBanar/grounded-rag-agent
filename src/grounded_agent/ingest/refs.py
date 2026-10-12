"""Find cross-references such as "Article 6(2)" or "Annex III" inside legal text.

References are resolved to unit ids of the same act, or of another act when the
text names it (for example "Article 9 of Regulation (EU) 2016/679" -> GDPR).
"""

from __future__ import annotations

import re

# Regulation numbers of acts we know, mapped to their document labels.
KNOWN_ACTS = {"2016/679": "GDPR", "2024/1689": "AI Act"}

_ART = re.compile(
    r"Articles?\s+(?P<num>\d+)(?P<para>(?:\(\d+\))?)"
    r"(?P<range>\s+(?:to|and)\s+\d+)?"
    r"(?P<of>\s+of\s+(?:Regulation|Directive)\s+\((?:EU|EC)\)\s+(?:No\s+)?(?P<act>\d{2,4}/\d+))?"
)
_ANNEX = re.compile(r"Annex\s+(?P<num>[IVXL]+)\b")
_ACT_ONLY = re.compile(r"Regulation\s+\(EU\)\s+(?P<act>2016/679|2024/1689)\b")


def find_refs(text: str, doc: str) -> list[str]:
    """Return candidate unit ids referenced in ``text``.

    Paragraph references become ``"<doc> Art. N(p)"``; plain article references
    and ranges become article ids ``"<doc> Art. N"``. Annexes become
    ``"<doc> Annex X"``. Ids may not exist; :func:`resolve` filters them.
    """
    found: list[str] = []

    def add(ref: str) -> None:
        if ref not in found:
            found.append(ref)

    for m in _ART.finditer(text):
        target = doc
        tail = text[m.end(): m.end() + 70]
        act = m.group("act")
        if not act:
            # "Article 4, point (4), of Regulation (EU) 2016/679": the act may follow a few words later
            mt = re.match(r"[^.;]{0,45}?\bof\s+(?:Regulation|Directive)\s+\((?:EU|EC|EEC)\)\s+(?:No\s+)?(\d{2,4}/\d+)", tail)
            if mt:
                act = mt.group(1)
            elif re.match(r"[^.;]{0,45}?\bof\s+(?:that|this)\s+Directive|[^.;]{0,45}?\bof\s+Directive", tail):
                continue  # article of a directive outside the corpus
        if act:
            if act not in KNOWN_ACTS:
                continue  # reference to an act outside the corpus
            target = KNOWN_ACTS[act]
        num = int(m.group("num"))
        para = m.group("para")
        add(f"{target} Art. {num}{para}" if para else f"{target} Art. {num}")
        rng = m.group("range")
        if rng and " to " in rng:
            end = int(rng.split()[-1])
            for n in range(num + 1, min(end, num + 30) + 1):
                add(f"{target} Art. {n}")
        elif rng:
            add(f"{target} Art. {int(rng.split()[-1])}")
    for m in _ANNEX.finditer(text):
        add(f"{doc} Annex {m.group('num')}")
    for m in _ACT_ONLY.finditer(text):
        other = KNOWN_ACTS[m.group("act")]
        if other != doc:
            add(other)  # whole-act reference, kept for follow_refs display only
    return found


def resolve(refs: list[str], known_ids: set[str], self_id: str, parent: str | None) -> list[str]:
    """Keep references that exist in the corpus and do not point to the unit itself."""
    out = []
    for r in refs:
        if r in known_ids and r != self_id and r != parent and r not in out:
            out.append(r)
    return out
