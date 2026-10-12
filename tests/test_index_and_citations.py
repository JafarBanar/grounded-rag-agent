from grounded_agent.citations import canon, check
from grounded_agent.index import tokenize


def test_tokenize_keeps_codes_and_splits_hyphens():
    toks = tokenize("High-risk systems under Regulation 2016/679")
    assert "high-risk" in toks and "high" in toks and "risk" in toks and "2016/679" in toks


def test_bm25_finds_recruitment(index):
    hits = index.search("recruitment of candidates", k=3)
    assert hits[0].unit.id.startswith("AI Act Annex III(2)")


def test_search_skips_whole_articles(index):
    assert all(h.unit.kind not in ("article", "annex") for h in index.search("high-risk", k=10))


def test_canon_forms():
    assert canon("AI Act Article 6(2)") == canon("ai act art. 6(2)") == canon("AI  Act Art 6(2)")


def test_check_passes_when_all_cited_and_retrieved(index):
    ans = "AI systems listed in Annex III are high-risk [AI Act Art. 6(2)]. Recruitment tools are listed there [AI Act Annex III(2)(a)]."
    rep = check(ans, index, {"AI Act Art. 6(2)", "AI Act Annex III(2)(a)"}, min_coverage=1.0)
    assert rep["ok"], rep["problems"]
    assert rep["cited"] == ["AI Act Art. 6(2)", "AI Act Annex III(2)(a)"]


def test_citation_with_abbreviation_is_not_split(index):
    # groundcite splits sentences at ". "; "Art. 6(2)" must not break the sentence
    rep = check("AI systems listed in Annex III are high-risk [AI Act Art. 6(2)].", index, {"AI Act Art. 6(2)"}, 1.0)
    assert rep["coverage"] == 1.0


def test_check_rejects_unretrieved_and_invented_citations(index):
    ans = "Recruitment tools are high-risk [AI Act Art. 6(2)]. Fines are huge [AI Act Art. 99(3)]."
    rep = check(ans, index, {"AI Act Art. 6(1)"}, min_coverage=1.0)
    assert not rep["ok"]
    assert "AI Act Art. 6(2)" in rep["unresolved"]
    assert any("do not exist" in p for p in rep["problems"])      # Art. 99 is not in the corpus


def test_reading_an_article_allows_its_paragraphs(index):
    rep = check("Annex III systems are high-risk [AI Act Art. 6(2)].", index, {"AI Act Art. 6"}, 1.0)
    assert rep["ok"]


def test_uncited_sentence_fails_coverage(index):
    ans = "Annex III systems are high-risk [AI Act Art. 6(2)]. They also need a conformity assessment before sale."
    assert not check(ans, index, {"AI Act Art. 6(2)"}, min_coverage=1.0)["ok"]
