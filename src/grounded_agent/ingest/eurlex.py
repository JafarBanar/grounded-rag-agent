"""Parse EU legal acts in the Publications Office XHTML format (Cellar / EUR-Lex).

Each act becomes citable units:

* article paragraphs  -> ``AI Act Art. 6(2)``     (from ids such as ``006.002``)
* definition points   -> ``AI Act Art. 3(1)``     (articles that are numbered lists)
* whole articles      -> ``AI Act Art. 6``        (always kept, used by read and refs)
* annex points        -> ``AI Act Annex III(4)``  (top-level numbered points)
* whole annexes       -> ``AI Act Annex III``
* recitals            -> ``AI Act Recital 53``
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from ..models import Unit
from .refs import find_refs

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_WS = re.compile(r"\s+")
_PARA_ID = re.compile(r"^(\d{3})\.(\d{3})$")
_POINT_START = re.compile(r"^\((\d+)\)\s")          # definition points "(1) 'AI system' means ..."
_ANNEX_POINT = re.compile(r"^(\d+)\.\s+\S")          # annex points "4. Employment, ..."
_LETTER = re.compile(r"(?:^|\s)\(([a-z])\)\s")       # lettered points "(a) ..."
LONG_UNIT = 0                                          # split any unit with lettered points (a), (b), ...


def _clean(text: str) -> str:
    return _WS.sub(" ", text.replace("\xa0", " ")).strip()


def _text(tag) -> str:
    return _clean(tag.get_text(" ", strip=True))


def _article_parts(art) -> tuple[str, str, list[tuple[str, str]], list[str]]:
    """Return (number, title, [(paragraph_number, text)], body_blocks) for one article div."""
    number = _clean(art.find(class_="oj-ti-art").get_text(" ", strip=True)).replace("Article", "").strip()
    sti = art.find(class_="oj-sti-art")
    title = _text(sti) if sti else ""
    paras = []
    for div in art.find_all(id=_PARA_ID, recursive=True):
        n = int(_PARA_ID.match(div["id"]).group(2))
        txt = _text(div)
        txt = re.sub(rf"^{n}\.\s+", "", txt)
        paras.append((str(n), txt))
    blocks = []
    if not paras:
        for el in art.find_all(["p", "table"], recursive=True):
            if el.get("class") and any(c in ("oj-ti-art", "oj-sti-art") for c in el["class"]):
                continue
            if el.name == "p" and el.find_parent("table") is not None:
                continue
            t = _text(el)
            if t:
                blocks.append(t)
    return number, title, paras, blocks


def _split_points(blocks: list[str]) -> list[tuple[str, str]]:
    """Group flat text blocks into numbered points "(n) ...", for definition lists."""
    points: list[tuple[str, str]] = []
    for b in blocks:
        m = _POINT_START.match(b)
        if m and (not points or int(m.group(1)) == int(points[-1][0]) + 1):
            points.append((m.group(1), b[m.end():]))
        elif points:
            points[-1] = (points[-1][0], points[-1][1] + " " + b)
        else:
            points.append(("0", b))
    return points


def split_letters(unit: Unit) -> list[Unit]:
    """Split a long unit with lettered points "(a) ... (b) ..." into child units "<id>(a)".

    Only consecutive letters starting at (a) count, so cross-references such as
    "point (c) of Article 5" in the middle of a sentence do not start a point.
    Each child keeps the unit's introductory sentence so it reads on its own.
    """
    if len(unit.text) <= LONG_UNIT:
        return []
    marks, expect = [], "a"
    for m in _LETTER.finditer(unit.text):
        if m.group(1) == expect:
            marks.append((m.group(1), m.start(1) - 1))
            expect = chr(ord(expect) + 1)
    if len(marks) < 2:
        return []
    intro = unit.text[: marks[0][1]].strip()
    children = []
    for i, (letter, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(unit.text)
        body = unit.text[start:end].strip()
        children.append(Unit(id=f"{unit.id}({letter})", doc=unit.doc, kind="point", title=unit.title,
                             text=f"{intro} {body}".strip()[:4000], parent=unit.id, source=unit.source))
    return children


def parse_act(path: Path, doc: str) -> list[Unit]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    units: list[Unit] = []
    src = path.name

    # Recitals
    for rct in soup.find_all(id=re.compile(r"^rct_\d+$")):
        n = rct["id"].split("_")[1]
        txt = re.sub(r"^\(\d+\)\s*", "", _text(rct))
        units.append(Unit(id=f"{doc} Recital {n}", doc=doc, kind="recital", title=f"Recital {n}", text=txt, source=src))

    # Articles
    for art in soup.find_all(id=re.compile(r"^art_\d+$")):
        number, title, paras, blocks = _article_parts(art)
        art_id = f"{doc} Art. {number}"
        if paras:
            full = " ".join(f"{n}. {t}" for n, t in paras)
            for n, t in paras:
                units.append(Unit(id=f"{art_id}({n})", doc=doc, kind="paragraph", title=title, text=t, parent=art_id, source=src))
        else:
            points = _split_points(blocks)
            numbered = [p for p in points if p[0] != "0"]
            if len(numbered) >= 3:
                intro = " ".join(t for n, t in points if n == "0")
                for n, t in numbered:
                    term = re.match(r"[‘'\"]([^’'\"]{2,80})[’'\"]\s+means", t)
                    ptitle = f"Definition of '{term.group(1)}'" if (term and title.lower().startswith("definition")) else title
                    units.append(Unit(id=f"{art_id}({n})", doc=doc, kind="point", title=ptitle,
                                      text=(f"{intro} ({n}) {t}" if intro else t), parent=art_id, source=src))
            full = " ".join(blocks)
        units.append(Unit(id=art_id, doc=doc, kind="article", title=title, text=full, source=src))

    # Annexes
    for anx in soup.find_all(id=re.compile(r"^anx_[IVXL]+$")):
        num = anx["id"].split("_")[1]
        anx_id = f"{doc} Annex {num}"
        blocks = [_text(p) for p in anx.find_all("p") if _text(p) and p.find_parent("table") is None]
        for tbl in anx.find_all("table"):
            t = _text(tbl)
            if t:
                blocks.append(t)
        title = blocks[1] if len(blocks) > 1 else anx_id
        points: list[tuple[str, str]] = []
        for b in blocks[2:]:
            m = _ANNEX_POINT.match(b)
            if m and (not points or int(m.group(1)) == int(points[-1][0]) + 1):
                points.append((m.group(1), b))
            elif points:
                points[-1] = (points[-1][0], points[-1][1] + " " + b)
        if len(points) >= 2:
            for n, t in points:
                units.append(Unit(id=f"{anx_id}({n})", doc=doc, kind="point", title=title, text=t, parent=anx_id, source=src))
        units.append(Unit(id=anx_id, doc=doc, kind="annex", title=title, text=" ".join(blocks[1:]), source=src))

    has_children = {u.parent for u in units if u.parent}
    extra: list[Unit] = []
    for u in units:
        if u.kind in ("paragraph", "point") or (u.kind == "article" and u.id not in has_children):
            extra.extend(split_letters(u))
    units.extend(extra)
    for u in units:
        u.refs = find_refs(u.text, doc)
    return units
