"""The tools the agent can call. Each returns short, citable text for the model."""

from __future__ import annotations

import re

from .citations import canon
from .index import HybridIndex

SNIPPET = 500


def _schema(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}}


class Toolbox:
    """Search, read and follow references over one index. Tracks what was retrieved."""

    def __init__(self, index: HybridIndex, embed_fn=None, docs: list[str] | None = None):
        self.index = index
        self.embed_fn = embed_fn
        self.docs = docs or sorted({u.doc for u in index.units})
        self.retrieved: set[str] = set()     # ids whose text was shown to the model
        self._canon = {canon(u.id): u.id for u in index.units}

    # ------------------------------------------------------------- schemas
    def schemas(self) -> list[dict]:
        docs = ", ".join(self.docs)
        return [
            _schema("search", f"Search the documents ({docs}) and return the best-matching provisions with their ids. "
                              "Use several short, specific queries for different parts of a question.",
                    {"query": {"type": "string", "description": "Search words, e.g. 'recruitment AI high-risk'"},
                     "doc": {"type": "string", "description": f"Optional: limit to one document: {docs}"}},
                    ["query"]),
            _schema("read", "Read the full text of one provision by its id (e.g. 'AI Act Art. 6(2)' or a whole article "
                            "'AI Act Art. 6'). Use it before relying on a provision.",
                    {"id": {"type": "string", "description": "Exact id as shown by search"}}, ["id"]),
            _schema("follow_refs", "List the provisions that a given provision refers to (articles, annexes, other acts), "
                                   "so you can read them next.",
                    {"id": {"type": "string", "description": "Exact id as shown by search or read"}}, ["id"]),
        ]

    # ------------------------------------------------------------- helpers
    def resolve_id(self, raw: str) -> str | None:
        r = raw.strip().strip("[]").strip()
        if r in self.index.by_id:
            return r
        c = canon(r)
        if c in self._canon:
            return self._canon[c]
        # "Article 6(2) AI Act" -> "AI Act Art. 6(2)"
        for d in self.docs:
            if d.lower() in c:
                rest = c.replace(d.lower(), "").strip(" ,")
                cand = canon(f"{d} {rest}")
                if cand in self._canon:
                    return self._canon[cand]
        return None

    def _line(self, uid: str, limit: int = SNIPPET) -> str:
        u = self.index.by_id[uid]
        text = u.text if len(u.text) <= limit else u.text[:limit].rsplit(" ", 1)[0] + " ..."
        return f"[{u.id}] {u.title}\n{text}"

    # ------------------------------------------------------------- tools
    def search(self, query: str, doc: str | None = None, k: int = 6) -> tuple[str, list[str]]:
        doc = doc if doc in self.docs else None
        hits = self.index.search(query, k=k, embed_fn=self.embed_fn, doc=doc)
        if not hits:
            return "No results. Try other words.", []
        ids = [h.unit.id for h in hits]
        self.retrieved.update(ids)
        return "\n\n".join(self._line(i) for i in ids), ids

    def read(self, id: str) -> tuple[str, list[str]]:
        uid = self.resolve_id(id)
        if uid is None:
            hits = self.index.search(id, k=3, embed_fn=self.embed_fn)
            return ("Unknown id '" + id + "'. Closest ids: " + ", ".join(h.unit.id for h in hits)), []
        u = self.index.by_id[uid]
        kids = self.index.children.get(uid, [])
        if u.kind in ("article", "annex") and kids:
            parts = [f"[{k}] {self.index.by_id[k].text}" for k in kids if self.index.by_id[k].parent == uid]
            body = "\n".join(parts)[:9000]
            shown = [uid] + kids
        else:
            body = u.text[:6000]
            shown = [uid]
        self.retrieved.update(shown)
        refs = f"\nRefers to: {', '.join(u.refs[:12])}" if u.refs else ""
        return f"[{u.id}] {u.title}\n{body}{refs}", shown

    def follow_refs(self, id: str) -> tuple[str, list[str]]:
        uid = self.resolve_id(id)
        if uid is None:
            return f"Unknown id '{id}'.", []
        u = self.index.by_id[uid]
        refs = list(u.refs)
        if u.parent:
            refs += [r for r in self.index.by_id[u.parent].refs if r not in refs] if u.parent in self.index.by_id else []
        if not refs:
            return f"{uid} does not refer to other provisions in this corpus.", []
        lines = [f"[{r}] {self.index.by_id[r].title}" for r in refs[:15] if r in self.index.by_id]
        return f"{uid} refers to:\n" + "\n".join(lines) + "\nUse read to see the text.", []

    def execute(self, name: str, args: dict) -> tuple[str, list[str]]:
        args = args or {}
        try:
            if name == "search":
                return self.search(str(args.get("query", "")), args.get("doc"))
            if name == "read":
                return self.read(str(args.get("id", "")))
            if name == "follow_refs":
                return self.follow_refs(str(args.get("id", "")))
            return f"Unknown tool '{name}'. Use search, read or follow_refs.", []
        except Exception as err:  # never crash the loop on a tool error
            return f"Tool error: {err}", []
