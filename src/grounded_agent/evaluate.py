"""Run the question set through the agent and the retrieve-once workflow, and score both."""

from __future__ import annotations

import json
import statistics as st
import time
from pathlib import Path

import yaml

from .agent import Agent, Result
from .baseline import Workflow
from .citations import canon


def _matches(cited: str, gold: str) -> bool:
    c, g = canon(cited), canon(gold)
    return c == g or c.startswith(g + "(") or g.startswith(c + "(")


def score(q: dict, res: Result) -> dict:
    out = {"id": q["id"], "type": q["type"], "status": res.status, "seconds": res.seconds,
           "model_calls": res.model_calls, "tool_calls": res.tool_calls,
           "tokens": res.prompt_tokens + res.output_tokens}
    drafts = [e for e in res.events if e["type"] == "validation"]
    out["first_draft_valid"] = bool(drafts and drafts[0]["ok"]) if drafts else None
    if q["type"] == "out":
        out["correct"] = res.status in ("not_found", "refused")
        return out
    groups = q.get("gold", [])
    cited_hits = sum(any(_matches(c, g) for c in res.cited for g in grp) for grp in groups)
    seen_hits = sum(any(_matches(r, g) for r in res.retrieved for g in grp) for grp in groups)
    out["answered"] = res.status == "answered"
    out["gold_cited"] = cited_hits / len(groups) if groups else 0.0
    out["gold_retrieved"] = seen_hits / len(groups) if groups else 0.0
    out["correct"] = out["answered"] and cited_hits == len(groups)
    return out


def summarize(rows: list[dict]) -> dict:
    inn = [r for r in rows if r["type"] != "out"]
    out = [r for r in rows if r["type"] == "out"]
    mean = lambda xs: round(st.mean(xs), 3) if xs else None
    s = {
        "questions": len(rows),
        "answered_in_scope": mean([r["answered"] for r in inn]),
        "fully_correct_in_scope": mean([r["correct"] for r in inn]),
        "gold_cited": mean([r["gold_cited"] for r in inn]),
        "gold_retrieved": mean([r["gold_retrieved"] for r in inn]),
        "correct_refusals_out_of_scope": mean([r["correct"] for r in out]),
        "first_draft_valid": mean([r["first_draft_valid"] for r in rows if r["first_draft_valid"] is not None]),
        "seconds_mean": mean([r["seconds"] for r in rows]),
        "seconds_p95": round(sorted(r["seconds"] for r in rows)[max(0, int(len(rows) * 0.95) - 1)], 2),
        "model_calls_mean": mean([r["model_calls"] for r in rows]),
        "tool_calls_mean": mean([r["tool_calls"] for r in rows]),
        "tokens_mean": mean([r["tokens"] for r in rows]),
    }
    by_type = {}
    for t in ("single", "multi", "cross"):
        rs = [r for r in inn if r["type"] == t]
        if rs:
            by_type[t] = {"n": len(rs), "fully_correct": mean([r["correct"] for r in rs]), "gold_cited": mean([r["gold_cited"] for r in rs])}
    s["by_type"] = by_type
    return s


def run_eval(cfg, questions_path: Path, out_dir: Path, modes=("agent", "workflow"), log=print) -> dict:
    qs = yaml.safe_load(Path(questions_path).read_text(encoding="utf-8"))["questions"]
    out_dir.mkdir(parents=True, exist_ok=True)
    systems = {}
    if "agent" in modes:
        systems["agent"] = Agent(cfg)
    if "workflow" in modes:
        systems["workflow"] = Workflow(cfg, index=systems["agent"].index if "agent" in systems else None)
    report = {"model": cfg.model, "embed_model": cfg.embed_model, "think": cfg.think,
              "date": time.strftime("%Y-%m-%d"), "systems": {}}
    for name, system in systems.items():
        rows, runs = [], []
        for q in qs:
            res = system.run(q["q"])
            row = score(q, res)
            rows.append(row)
            runs.append(res.to_dict())
            log(f"[{name}] {q['id']:4} {row['status']:9} correct={row['correct']!s:5} {res.seconds:6.1f}s cited={res.cited[:4]}")
        report["systems"][name] = {"summary": summarize(rows), "rows": rows}
        (out_dir / f"runs_{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in runs), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def to_markdown(report: dict) -> str:
    names = list(report["systems"])
    keys = [("fully_correct_in_scope", "In-scope questions fully correct (all gold provisions cited)"),
            ("gold_cited", "Gold provisions cited (mean share)"),
            ("gold_retrieved", "Gold provisions retrieved (mean share)"),
            ("answered_in_scope", "In-scope questions answered (not refused)"),
            ("correct_refusals_out_of_scope", "Out-of-scope questions correctly refused"),
            ("first_draft_valid", "First draft passed the citation check"),
            ("seconds_mean", "Seconds per question (mean)"), ("seconds_p95", "Seconds per question (95th percentile)"),
            ("model_calls_mean", "Model calls per question"), ("tool_calls_mean", "Tool calls per question"),
            ("tokens_mean", "Tokens per question")]
    lines = [f"Model `{report['model']}`, embeddings `{report['embed_model']}`, thinking {report['think']}, run {report['date']}.", "",
             "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k, label in keys:
        lines.append(f"| {label} | " + " | ".join(str(report['systems'][n]['summary'].get(k)) for n in names) + " |")
    lines += ["", "| Question type | " + " | ".join(f"{n}: fully correct" for n in names) + " |", "|---|" + "---|" * len(names)]
    for t in ("single", "multi", "cross"):
        cells = []
        for n in names:
            bt = report["systems"][n]["summary"]["by_type"].get(t)
            cells.append(f"{bt['fully_correct']} (n={bt['n']})" if bt else "-")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
