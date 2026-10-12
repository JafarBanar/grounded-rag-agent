"""Command line: build an index, ask a question, run the evaluation, start the web app or MCP server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT, CorpusConfig


def _cfg(name_or_path: str) -> CorpusConfig:
    p = Path(name_or_path)
    if not p.exists():
        p = ROOT / "corpora" / f"{name_or_path}.yaml"
    return CorpusConfig.load(p)


def cmd_build(a):
    from .build import build
    build(_cfg(a.corpus), embed=not a.no_embed)


def cmd_ask(a):
    from .agent import Agent, save_trace
    from .baseline import Workflow
    cfg = _cfg(a.corpus)
    if a.model:
        cfg.model = a.model
    cfg.think = a.think or cfg.think
    cfg.verify = a.verify or cfg.verify
    system = Workflow(cfg) if a.workflow else Agent(cfg)

    def show(ev):
        if a.quiet:
            return
        if ev["type"] == "tool_call":
            print(f"  [{ev['t']:5.1f}s] {ev['name']}({json.dumps(ev['args'], ensure_ascii=False)})", file=sys.stderr)
        elif ev["type"] == "tool_result":
            print(f"           -> {', '.join(ev['ids'][:6]) or '(no units)'}", file=sys.stderr)
        elif ev["type"] == "plan":
            print(f"  [{ev['t']:5.1f}s] plan: {'; '.join(ev['queries'])}", file=sys.stderr)
        elif ev["type"] == "verify":
            print(f"  [{ev['t']:5.1f}s] claim check: {'passed' if ev['ok'] else 'failed: ' + '; '.join(ev['problems'])}", file=sys.stderr)
        elif ev["type"] in ("nudge", "repair"):
            print(f"  [{ev['t']:5.1f}s] {ev['type']}: {'; '.join(ev.get('problems', [])) or 'search first'}", file=sys.stderr)

    res = system.run(a.question, on_event=show)
    print(f"\n{res.answer}\n")
    print(f"status: {res.status} | cited: {', '.join(res.cited) or '-'} | {res.seconds}s, "
          f"{res.model_calls} model calls, {res.tool_calls} tool calls")
    save_trace(res, ROOT / "data" / "traces" / cfg.name)


def cmd_eval(a):
    from .evaluate import run_eval
    cfg = _cfg(a.corpus)
    if a.model:
        cfg.model = a.model
    cfg.think = a.think or cfg.think
    cfg.verify = a.verify or cfg.verify
    out = Path(a.out or ROOT / "eval" / "results" / cfg.name)
    rep = run_eval(cfg, Path(a.questions), out, modes=tuple(a.modes.split(",")))
    print((out / "report.md").read_text())


def cmd_serve(a):
    import uvicorn
    from .server import create_app
    uvicorn.run(create_app(a.corpus.split(",")), host=a.host, port=a.port)


def cmd_mcp(a):
    from .mcp_server import run
    run(a.corpus)


def main(argv=None):
    p = argparse.ArgumentParser(prog="gra", description="grounded-rag-agent: cite every claim or refuse")
    sub = p.add_subparsers(required=True)
    b = sub.add_parser("build", help="parse sources, embed and save the index")
    b.add_argument("corpus"); b.add_argument("--no-embed", action="store_true"); b.set_defaults(fn=cmd_build)
    q = sub.add_parser("ask", help="ask one question")
    q.add_argument("corpus"); q.add_argument("question"); q.add_argument("--workflow", action="store_true", help="use retrieve-once RAG instead of the agent")
    q.add_argument("--model"); q.add_argument("--think", action="store_true"); q.add_argument("--quiet", action="store_true")
    q.add_argument("--verify", action="store_true", help="also check each cited sentence against its source text"); q.set_defaults(fn=cmd_ask)
    e = sub.add_parser("eval", help="run the evaluation set")
    e.add_argument("corpus"); e.add_argument("--questions", default=str(ROOT / "eval" / "questions_eu.yaml"))
    e.add_argument("--modes", default="agent,workflow"); e.add_argument("--model"); e.add_argument("--think", action="store_true"); e.add_argument("--out")
    e.add_argument("--verify", action="store_true")
    e.set_defaults(fn=cmd_eval)
    s = sub.add_parser("serve", help="web UI and API")
    s.add_argument("--corpus", default="eu_ai_law"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000); s.set_defaults(fn=cmd_serve)
    m = sub.add_parser("mcp", help="run the tools as an MCP server over stdio")
    m.add_argument("corpus"); m.set_defaults(fn=cmd_mcp)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
