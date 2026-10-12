"""Load private documents (Markdown, text, PDF, Word) into citable units.

Citation ids are readable and stable:

* Markdown / text: ``master-cv.md §Experience``         (one unit per heading section)
* Word:            ``Project_Book.docx §QRTech: LLM platform``
* PDF:             ``ispcs2024.pdf p.3``               (one unit per page)

Long sections are split into parts ``§Heading #2`` so every unit stays small enough to cite.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Unit

MAX_CHARS = 1800
_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")


def _split_long(base_id: str, doc: str, kind: str, title: str, text: str, source: str) -> list[Unit]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= MAX_CHARS:
        return [Unit(id=base_id, doc=doc, kind=kind, title=title, text=text, source=source)]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(cur) + len(sent) > MAX_CHARS and cur:
            parts.append(cur)
            cur = ""
        cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return [Unit(id=f"{base_id} #{i}", doc=doc, kind=kind, title=title, text=p, source=source)
            for i, p in enumerate(parts, 1)]


def _unique(units: list[Unit]) -> list[Unit]:
    seen: dict[str, int] = {}
    for u in units:
        if u.id in seen:
            seen[u.id] += 1
            u.id = f"{u.id} ({seen[u.id]})"
        else:
            seen[u.id] = 1
    return units


def load_markdown(path: Path, doc: str) -> list[Unit]:
    units, heading, buf = [], "Start", []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _HEADING.match(line)
        if m:
            units += _split_long(f"{path.name} §{heading}", doc, "section", heading, "\n".join(buf), path.name)
            heading, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    units += _split_long(f"{path.name} §{heading}", doc, "section", heading, "\n".join(buf), path.name)
    return units


def load_docx(path: Path, doc: str) -> list[Unit]:
    """Walk the Word body in document order, so tables stay in the section where they appear."""
    import docx  # python-docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    units, heading, buf = [], "Start", []
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            style = (p.style.name or "").lower() if p.style is not None else ""
            if style.startswith("heading") or style == "title":
                units += _split_long(f"{path.name} §{heading}", doc, "section", heading, " ".join(buf), path.name)
                heading, buf = p.text.strip() or heading, []
            elif p.text.strip():
                buf.append(p.text.strip())
        elif tag == "tbl":
            t = Table(child, d)
            rows = []
            for r in t.rows:
                cells = []
                for c in r.cells:
                    txt = c.text.strip()
                    if txt and (not cells or cells[-1] != txt):   # merged cells repeat their text
                        cells.append(txt)
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                buf.append(" ; ".join(rows))
    units += _split_long(f"{path.name} §{heading}", doc, "section", heading, " ".join(buf), path.name)
    return units


def load_pdf(path: Path, doc: str) -> list[Unit]:
    from pypdf import PdfReader
    units = []
    for i, page in enumerate(PdfReader(str(path)).pages, 1):
        text = page.extract_text() or ""
        units += _split_long(f"{path.name} p.{i}", doc, "page", f"{path.stem}, page {i}", text, path.name)
    return units


LOADERS = {".md": load_markdown, ".markdown": load_markdown, ".txt": load_markdown,
           ".docx": load_docx, ".pdf": load_pdf}


def load_path(path: Path, doc: str) -> list[Unit]:
    """Load one file, or every supported file in a folder (not recursive into hidden folders)."""
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.parts))
    units: list[Unit] = []
    for f in files:
        loader = LOADERS.get(f.suffix.lower())
        if loader:
            units += loader(f, doc)
    return _unique(units)
