from grounded_agent.ingest.refs import find_refs


def test_article_paragraphs_become_units(units):
    ids = {u.id for u in units}
    assert {"AI Act Art. 6", "AI Act Art. 6(1)", "AI Act Art. 6(2)"} <= ids
    p = next(u for u in units if u.id == "AI Act Art. 6(2)")
    assert p.parent == "AI Act Art. 6"
    assert p.title == "Classification rules for high-risk AI systems"
    assert p.text.startswith("AI systems referred to in Annex III")   # paragraph number stripped


def test_definition_lists_split_into_points(units):
    d = {u.id: u for u in units}
    assert "AI Act Art. 3(1)" in d and "AI Act Art. 3(3)" in d
    assert "machine-based system" in d["AI Act Art. 3(1)"].text
    assert d["AI Act Art. 3(1)"].text.startswith("For the purposes")     # intro kept for context


def test_annex_points_and_lettered_points(units):
    d = {u.id: u for u in units}
    assert "AI Act Annex III(1)" in d and "AI Act Annex III(2)" in d
    assert "AI Act Annex III(2)(a)" in d and "AI Act Annex III(2)(b)" in d
    assert d["AI Act Annex III(2)(b)"].parent == "AI Act Annex III(2)"


def test_recitals(units):
    assert any(u.id == "AI Act Recital 1" and u.kind == "recital" for u in units)


def test_references_resolve_inside_corpus(units):
    d = {u.id: u for u in units}
    assert d["AI Act Art. 6(2)"].refs == ["AI Act Annex III"]          # Article 9 is not in the fixture


def test_find_refs_maps_other_acts():
    assert find_refs("as defined in Article 4, point (1), of Regulation (EU) 2016/679", "AI Act")[0] == "GDPR Art. 4"
    assert find_refs("Article 9(1) of Regulation (EU) 2016/679", "AI Act")[0] == "GDPR Art. 9(1)"
    assert find_refs("Articles 9 to 11 apply", "AI Act") == ["AI Act Art. 9", "AI Act Art. 10", "AI Act Art. 11"]
    assert find_refs("Article 5 of Directive (EU) 2019/882", "AI Act") == []          # act outside the corpus
