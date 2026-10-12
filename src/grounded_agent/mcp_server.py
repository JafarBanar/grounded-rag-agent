"""Expose the corpus tools (and the full agent) as an MCP server, for Claude Desktop, IDEs or other agents."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .agent import Agent
from .config import ROOT, CorpusConfig
from .tools import Toolbox


def build_server(corpus: str) -> MCPServer:
    cfg = CorpusConfig.load(ROOT / "corpora" / f"{corpus}.yaml")
    agent = Agent(cfg)
    box = Toolbox(agent.index, embed_fn=agent.client.embed)
    mcp = MCPServer(f"grounded-rag-agent ({cfg.name})", instructions="Search, read and cite EU legal provisions; every answer is citation-checked.")

    @mcp.tool()
    def search(query: str, doc: str | None = None) -> str:
        """Search the documents and return matching provisions with their citation ids."""
        return box.search(query, doc)[0]

    @mcp.tool()
    def read(id: str) -> str:
        """Read the full text of one provision or article by its citation id."""
        return box.read(id)[0]

    @mcp.tool()
    def follow_refs(id: str) -> str:
        """List the provisions that a provision refers to."""
        return box.follow_refs(id)[0]

    @mcp.tool()
    def ask(question: str) -> str:
        """Run the full local agent: search, read, answer with checked citations, or refuse."""
        res = agent.run(question)
        return f"{res.answer}\n\nstatus: {res.status}; cited: {', '.join(res.cited) or '-'}"

    return mcp


def run(corpus: str) -> None:
    build_server(corpus).run()   # stdio transport
