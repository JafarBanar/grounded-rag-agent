"""The agent loop with a scripted fake model: no Ollama needed."""

from grounded_agent.agent import Agent
from grounded_agent.llm import ChatResult


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


class FakeClient:
    """Returns scripted replies in order; embed is unused (BM25 only)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.embed = None

    def chat(self, messages, tools=None, format=None):
        self.seen.append(messages[-1])
        r = self.replies.pop(0)
        if isinstance(r, list):
            return ChatResult(content="", tool_calls=r, prompt_tokens=10, output_tokens=5)
        return ChatResult(content=r, tool_calls=[], prompt_tokens=10, output_tokens=5)


def run(cfg, index, replies):
    agent = Agent(cfg, index=index, client=FakeClient(replies))
    return agent.run("Is a recruitment AI high-risk?"), agent.client


def test_search_read_answer(cfg, index):
    res, _ = run(cfg, index, [
        [call("search", query="recruitment")],
        [call("read", id="AI Act Art. 6")],
        "Recruitment AI is listed in Annex III [AI Act Annex III(2)(a)]. Annex III systems are high-risk [AI Act Art. 6(2)].",
    ])
    assert res.status == "answered"
    assert res.cited == ["AI Act Annex III(2)(a)", "AI Act Art. 6(2)"]
    assert res.tool_calls == 2 and res.model_calls == 3


def test_answer_without_search_gets_nudged(cfg, index):
    res, client = run(cfg, index, [
        "It is high-risk.",
        [call("search", query="recruitment high-risk")],
        "Recruitment AI is listed in Annex III [AI Act Annex III(2)(a)].",
    ])
    assert res.status == "answered"
    assert any(e["type"] == "nudge" for e in res.events)


def test_bad_citation_is_repaired_then_refused(cfg, index):
    res, client = run(cfg, index, [
        [call("search", query="recruitment")],
        "Recruitment AI is high-risk [AI Act Art. 77].",
        "Recruitment AI is high-risk [AI Act Art. 77].",
    ])
    assert res.status == "refused"
    assert "not answering" in res.answer
    assert "failed the citation check" in client.seen[2]["content"]


def test_not_found_is_passed_through(cfg, index):
    res, _ = run(cfg, index, [[call("search", query="minimum wage")], "NOT FOUND: the act does not cover wages."])
    assert res.status == "not_found"


def test_unknown_id_suggests_alternatives(cfg, index):
    from grounded_agent.tools import Toolbox
    text, shown = Toolbox(index).read("AI Act Art. 77")
    assert text.startswith("Unknown id") and shown == []
    assert Toolbox(index).resolve_id("Article 6(2) AI Act") == "AI Act Art. 6(2)"


def test_plan_step_adds_search_plan(cfg, index):
    cfg.plan = True
    res, client = run(cfg, index, [
        '{"queries": ["recruitment AI high-risk", "classification rules"]}',
        [call("search", query="recruitment AI high-risk")],
        "Recruitment AI is listed in Annex III [AI Act Annex III(2)(a)].",
    ])
    assert res.status == "answered"
    assert any(e["type"] == "plan" and len(e["queries"]) == 2 for e in res.events)
    assert "Search plan" in client.seen[1]["content"]
