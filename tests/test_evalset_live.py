from tests.conftest import LIVE_FIXTURES, RETRIEVAL_FIXTURES

MIN_LIVE_GOLD_RECALL = 18

MAX_LIVE_UNRELATED_ADMITTED = 1


def _gate(rag, manifest, fixture, question):
    rag.current_url = manifest[fixture]['url']
    return rag.retrieve(question)


def test_the_live_set_is_disjoint_from_the_tuned_set(evalset, evalset_live):
    tuned = {row['fixture'] for row in evalset['answerable']}
    live = {row['fixture'] for row in evalset_live['answerable']}
    assert not tuned & live
    assert not set(RETRIEVAL_FIXTURES) & set(LIVE_FIXTURES)
    assert live == set(LIVE_FIXTURES)


def test_every_live_answer_key_is_present_in_its_article(articles, evalset_live):
    missing = [
        (row['fixture'], row['answer_key'])
        for row in evalset_live['answerable']
        if row['answer_key']
        not in articles[row['fixture']]['title'] + "\n" + articles[row['fixture']]['content']
    ]
    assert not missing


def test_gold_chunk_recall_on_unseen_articles_meets_the_measured_budget(
    indexed_rag_live, manifest, evalset_live
):
    hits = [
        row
        for row in evalset_live['answerable']
        if any(
            r.kept and row['answer_key'] in r.document
            for r in _gate(indexed_rag_live, manifest, row['fixture'], row['question'])
        )
    ]
    assert len(hits) >= MIN_LIVE_GOLD_RECALL, (
        f"{len(hits)} از {len(evalset_live['answerable'])} — "
        f"افت نسبت به بودجهٔ سنجیده‌شده روی مقاله‌های دیده‌نشده"
    )


def test_unrelated_questions_are_rejected_on_unseen_articles(
    indexed_rag_live, manifest, evalset_live
):
    admitted = [
        (name, question)
        for question in evalset_live['unrelated']
        for name in LIVE_FIXTURES
        if any(c.kept for c in _gate(indexed_rag_live, manifest, name, question))
    ]
    assert len(admitted) <= MAX_LIVE_UNRELATED_ADMITTED, (
        f"{len(admitted)} از {len(evalset_live['unrelated']) * len(LIVE_FIXTURES)} — "
        f"{admitted}"
    )


def test_live_misses_are_ranking_misses_not_gate_misses(
    indexed_rag_live, manifest, evalset_live
):
    gate_misses = []
    for row in evalset_live['answerable']:
        candidates = _gate(indexed_rag_live, manifest, row['fixture'], row['question'])
        if any(c.kept and row['answer_key'] in c.document for c in candidates):
            continue
        if any(row['answer_key'] in c.document for c in candidates):
            gate_misses.append(row['question'])
    assert not gate_misses


def test_the_shipped_budget_does_not_bind_on_the_unseen_set_either(
    indexed_rag_live, manifest, evalset_live
):
    for row in evalset_live['answerable']:
        candidates = _gate(indexed_rag_live, manifest, row['fixture'], row['question'])
        kept = [c.document for c in candidates if c.kept]
        assert kept == [c.document for c in candidates if c.in_prompt], row['question']
