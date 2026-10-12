"""The agent: a tool-calling loop over one corpus, with budgets, citation checks and refusal.

Flow (see docs/architecture.md):

    question -> [model decides: search / read / follow_refs] x N -> draft answer
             -> citation check (groundcite + evidence ledger)
             -> pass: answer | fail: one repair round with the problems listed | still failing: refuse
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .citations import BRACKET, check, expand_multi_citations, protect_citations, PROTECT
from .config import CorpusConfig
from .index import HybridIndex
from .llm import OllamaClient
from .tools import Toolbox

NOT_FOUND = "NOT FOUND:"

SYSTEM = """You are a careful research assistant. The documents are: {description}

Rules:
1. Use the tools to find evidence before you answer. Search with short, specific queries. When a search result is one paragraph or point of an article that looks relevant, read the whole article (for example read "AI Act Art. 50") so you see all of it. Use follow_refs when a provision points to another one (for example an annex or another article).
2. Before you answer, check that the retrieved text directly answers every part of the question. If a part is not directly answered, search again with different words (synonyms, the legal term, the article topic) before giving up. Questions with two parts usually need at least two searches.
3. Answer only from text the tools returned in this conversation. Never answer from memory.
4. End every factual sentence with the id of its source in square brackets, exactly as the tools show it, for example {citation_example}. Put one id in each bracket; for two sources write [id one][id two].
5. If, after searching, the documents do not answer the question at all, reply with "{not_found}" followed by one sentence saying what you searched for. If they answer it only partly, give the cited part and then one final sentence that starts with "Not shown in the documents:" naming what is missing; that sentence needs no citation.
6. Be concise and direct: a short answer first, then the supporting points. No headings.
{extra}"""

Event = dict

PLAN_PROMPT = """Split this question into the separate things that must be looked up in the documents ({docs}).
Return 1 to 4 short search queries, one per thing, using the words the law itself would likely use.
Question: {question}"""
VERIFY_PROMPT = """You are checking an answer against its sources. For each numbered claim, decide whether the
quoted source text explicitly supports it. Be strict: a claim is NOT supported if it changes who did something,
adds a number, date, tool or result that is not in the source, or overstates the source. Paraphrase is fine.

{claims}"""
VERIFY_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object",
                 "properties": {"claim": {"type": "integer"}, "supported": {"type": "boolean"}, "reason": {"type": "string"}},
                 "required": ["claim", "supported", "reason"]}}}, "required": ["verdicts"]}

PLAN_SCHEMA = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"},
               "minItems": 1, "maxItems": 4}}, "required": ["queries"]}


@dataclass
class Result:
    question: str
    answer: str
    status: str                     # answered | not_found | refused | error
    cited: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    draft: str = ""
    model_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    events: list[Event] = field(default_factory=list)
    mode: str = "agent"

    def to_dict(self) -> dict:
        return asdict(self)


def system_prompt(cfg: CorpusConfig) -> str:
    extra = "\n".join(f"{i}. {r}" for i, r in enumerate(cfg.extra_rules, 7))
    return SYSTEM.format(description=cfg.description.strip(), citation_example=cfg.citation_example,
                         not_found=NOT_FOUND, extra=extra)


def refusal_text(retrieved: list[str]) -> str:
    seen = ", ".join(retrieved[:8]) if retrieved else "nothing relevant"
    return ("I could not give an answer in which every claim is supported by a retrieved provision, "
            f"so I am not answering. Provisions checked: {seen}.")


class Agent:
    def __init__(self, cfg: CorpusConfig, index: HybridIndex | None = None, client: OllamaClient | None = None):
        self.cfg = cfg
        self.index = index or HybridIndex.load(Path(cfg.index_dir))
        self.client = client or OllamaClient(cfg.model, cfg.embed_model, think=cfg.think)

    def run(self, question: str, on_event: Callable[[Event], None] | None = None) -> Result:
        t0 = time.perf_counter()
        tools = Toolbox(self.index, embed_fn=self.client.embed)
        res = Result(question=question, answer="", status="error")

        def emit(ev: Event) -> None:
            ev["t"] = round(time.perf_counter() - t0, 2)
            res.events.append(ev)
            if on_event:
                on_event(ev)

        user_msg = question
        if self.cfg.plan:
            plan = self.plan(question, tools.docs)
            res.model_calls += 1
            if plan:
                emit({"type": "plan", "queries": plan})
                user_msg = (f"{question}\n\nSearch plan (search each of these, then read what you rely on): "
                            + "; ".join(f'"{q}"' for q in plan))
        messages = [{"role": "system", "content": system_prompt(self.cfg)},
                    {"role": "user", "content": user_msg}]
        repairs_left = self.cfg.repair_attempts
        answer = ""
        nudged = False
        for step in range(self.cfg.max_steps + self.cfg.repair_attempts * 3):
            budget_left = self.cfg.max_tool_calls - res.tool_calls
            if budget_left <= 0:
                messages.append({"role": "user", "content": "Tool budget used. Answer now from what you have retrieved, with citations."})
            r = self.client.chat(messages, tools.schemas() if budget_left > 0 else None)
            res.model_calls += 1
            res.prompt_tokens += r.prompt_tokens
            res.output_tokens += r.output_tokens
            assistant = {"role": "assistant", "content": r.content}
            if r.tool_calls:
                assistant["tool_calls"] = r.tool_calls
            messages.append(assistant)

            if r.tool_calls:
                for call in r.tool_calls[: max(budget_left, 0)]:
                    fn = call.get("function", {})
                    name, args = fn.get("name", ""), fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    emit({"type": "tool_call", "name": name, "args": args})
                    text, shown = tools.execute(name, args)
                    res.tool_calls += 1
                    emit({"type": "tool_result", "name": name, "ids": shown, "chars": len(text)})
                    messages.append({"role": "tool", "tool_name": name, "content": text})
                continue

            answer = r.content.strip()
            emit({"type": "draft", "text": answer})
            if res.tool_calls == 0 and not nudged:
                nudged = True
                messages.append({"role": "user", "content": "You have not used the tools yet. Search the documents first, "
                                 "then answer only from what they return, with citations."})
                emit({"type": "nudge"})
                continue
            if answer.upper().startswith(NOT_FOUND):
                res.status = "not_found"
                break
            report = check(answer, self.index, tools.retrieved, self.cfg.min_coverage)
            emit({"type": "validation", "ok": report["ok"], "problems": report["problems"]})
            res.validation = report
            if report["ok"] and self.cfg.verify:
                res.model_calls += 1
                bad = self.verify(answer)
                emit({"type": "verify", "ok": not bad, "problems": bad})
                if bad:
                    report = dict(report, ok=False, problems=[f"not supported by its cited source: {b}" for b in bad])
                    res.validation = report
            if report["ok"]:
                res.status = "answered"
                res.cited = report["cited"]
                break
            if repairs_left > 0:
                repairs_left -= 1
                feedback = ("Your answer failed the citation check: " + "; ".join(report["problems"]) +
                            ". Fix it: cite only ids that the tools returned (search or read more if needed), "
                            "put a citation on every factual sentence, and remove claims you cannot support. "
                            f"If you cannot support an answer, reply with {NOT_FOUND}")
                messages.append({"role": "user", "content": feedback})
                emit({"type": "repair", "problems": report["problems"]})
                continue
            res.status = "refused"
            break
        else:
            res.status = "refused"

        res.draft = answer
        res.retrieved = sorted(tools.retrieved)
        if res.status == "answered":
            res.answer = expand_multi_citations(answer, sorted({u.doc for u in self.index.units}))
        elif res.status == "not_found":
            res.answer = answer
        else:
            res.answer = refusal_text(res.retrieved)
        res.seconds = round(time.perf_counter() - t0, 2)
        emit({"type": "final", "status": res.status})
        return res


def _plan(self, question: str, docs: list[str]) -> list[str]:
    """Plan-and-execute: ask for the sub-questions as structured JSON before the tool loop."""
    try:
        r = self.client.chat([{"role": "user", "content": PLAN_PROMPT.format(docs=", ".join(docs), question=question)}],
                             format=PLAN_SCHEMA)
        queries = json.loads(r.content).get("queries", [])
        return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:4]
    except Exception:
        return []


Agent.plan = _plan


def _verify(self, answer: str, max_claims: int = 12) -> list[str]:
    """Claim check: ask the model, with structured output, whether each cited sentence is supported by its source text."""
    from groundcite import factual_sentences
    docs = sorted({u.doc for u in self.index.units})
    text = protect_citations(expand_multi_citations(answer, docs))
    claims = []
    for sent in factual_sentences(text):
        ids = [i for i in (self._resolve_cited(r) for r in BRACKET.find(sent)) if i]
        if ids:
            claims.append((sent.replace(PROTECT, ". "), ids))
    if not claims:
        return []
    blocks = []
    for n, (sent, ids) in enumerate(claims[:max_claims], 1):
        sources = " / ".join(f"[{i}] {self._source_text(i)}" for i in ids[:2])
        blocks.append(f"Claim {n}: {sent}\nSource: {sources}")
    try:
        r = self.client.chat([{"role": "user", "content": VERIFY_PROMPT.format(claims="\n\n".join(blocks))}], format=VERIFY_SCHEMA)
        verdicts = json.loads(r.content).get("verdicts", [])
    except Exception:
        return []          # a failed verifier call never blocks an answer that passed the citation check
    bad = []
    for v in verdicts:
        n = v.get("claim")
        if isinstance(n, int) and 1 <= n <= len(blocks) and v.get("supported") is False:
            bad.append(f'"{claims[n - 1][0][:160]}" ({v.get("reason", "")[:160]})')
    return bad


def _resolve_cited(self, ref: str):
    from .citations import canon
    if not hasattr(self, "_canon_ids"):
        self._canon_ids = {canon(u.id): u.id for u in self.index.units}
    return self._canon_ids.get(canon(ref))


def _source_text(self, uid: str, limit: int = 1500) -> str:
    u = self.index.by_id[uid]
    kids = self.index.children.get(uid, [])
    text = u.text if not kids or u.text else " ".join(self.index.by_id[k].text for k in kids)
    return text[:limit]


Agent.verify = _verify
Agent._resolve_cited = _resolve_cited
Agent._source_text = _source_text


def save_trace(res: Result, folder: Path) -> Path:
    """Append one run to a JSONL trace file (one line per question)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
    return path
