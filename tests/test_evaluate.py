from grounded_agent.agent import Result
from grounded_agent.evaluate import _matches, score, summarize


def test_gold_matching_is_containment_based():
    assert _matches("AI Act Art. 9(2)", "AI Act Art. 9")
    assert _matches("AI Act Art. 6", "AI Act Art. 6(3)")
    assert not _matches("AI Act Art. 99(3)", "AI Act Art. 9")


def test_score_and_summary():
    q_in = {"id": "m1", "type": "multi", "gold": [["AI Act Annex III(3)"], ["AI Act Art. 9"]]}
    r1 = Result(question="", answer="", status="answered", cited=["AI Act Annex III(3)(b)", "AI Act Art. 9(1)"],
                retrieved=["AI Act Annex III(3)(b)", "AI Act Art. 9(1)"], seconds=2.0)
    q_out = {"id": "o1", "type": "out", "expect": "not_found"}
    r2 = Result(question="", answer="", status="not_found", seconds=1.0)
    rows = [score(q_in, r1), score(q_out, r2)]
    assert rows[0]["correct"] and rows[0]["gold_cited"] == 1.0
    assert rows[1]["correct"]
    s = summarize(rows)
    assert s["fully_correct_in_scope"] == 1.0 and s["correct_refusals_out_of_scope"] == 1.0
