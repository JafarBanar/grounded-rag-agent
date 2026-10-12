"""Web UI and HTTP API. The agent's steps stream to the browser as server-sent events."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .agent import Agent, save_trace
from .baseline import Workflow
from .config import ROOT, CorpusConfig

WEB = ROOT / "web"


class AskRequest(BaseModel):
    question: str
    corpus: str = "eu_ai_law"
    mode: str = "agent"          # agent | workflow


def create_app(corpora: list[str]) -> FastAPI:
    app = FastAPI(title="grounded-rag-agent")
    agents: dict[str, Agent] = {}
    for name in corpora:
        path = ROOT / "corpora" / f"{name}.yaml"
        if path.exists() and (CorpusConfig.load(path).index_dir / "units.jsonl").exists():
            agents[name] = Agent(CorpusConfig.load(path))
    if not agents:
        raise RuntimeError("no built corpus found; run `gra build eu_ai_law` first")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/corpora")
    def list_corpora():
        return [{"name": n, "description": a.cfg.description, "model": a.cfg.model,
                 "units": len(a.index.units)} for n, a in agents.items()]

    @app.get("/api/unit")
    def unit(corpus: str, id: str):
        a = agents.get(corpus)
        u = a.index.by_id.get(id) if a else None
        if not u:
            raise HTTPException(404, "unknown unit")
        return {"id": u.id, "title": u.title, "text": u.text, "refs": u.refs, "parent": u.parent}

    @app.post("/api/ask")
    def ask(req: AskRequest):
        a = agents.get(req.corpus)
        if not a:
            raise HTTPException(404, "unknown corpus")
        system = a if req.mode == "agent" else Workflow(a.cfg, index=a.index, client=a.client)
        q: queue.Queue = queue.Queue()

        def work():
            try:
                res = system.run(req.question, on_event=lambda ev: q.put(("event", ev)))
                save_trace(res, ROOT / "data" / "traces" / a.cfg.name)
                q.put(("result", res.to_dict()))
            except Exception as err:  # report errors to the page instead of hanging
                q.put(("error", {"message": str(err)}))
            q.put(("done", None))

        threading.Thread(target=work, daemon=True).start()

        def stream():
            while True:
                kind, payload = q.get()
                if kind == "done":
                    break
                yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
