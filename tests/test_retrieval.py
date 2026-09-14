import pytest

import news_qa.rag
from news_qa.normalize import normalize
from news_qa.policy import GROUNDING_CLAUSE, OPEN, STRICT
from news_qa.rag import NO_CONTEXT_MESSAGE, RELEVANCE_THRESHOLD
from tests.conftest import RETRIEVAL_FIXTURES

IN_ARTICLE = {
    'varzesh3_2410970': ['سرمربی جدید تراکتور کیست؟', 'تراکتور تقریبا چند دیدار دارد؟'],
    'mehr_6914285': ['سیره پیامبر چه ویژگی‌هایی داشت؟', 'دنیای مدرن چه مشکلی ایجاد کرده است؟'],
}

UNRELATED = ['قیمت دلار چند است؟', 'پایتخت ژاپن کجاست؟', 'دستور پخت قرمه‌سبزی چیست؟']


def test_reingesting_the_same_article_does_not_duplicate(indexed_rag, articles, manifest):
    name = 'varzesh3_2410970'
    before = indexed_rag.collection.count()
    for _ in range(3):
        indexed_rag.chunk_and_store(
            articles[name]['content'],
            {'title': articles[name]['title'], 'url': manifest[name]['url']},
        )
    assert indexed_rag.collection.count() == before


def test_chunk_ids_are_deterministic(indexed_rag, articles, manifest):
    name = 'mehr_6914285'
    url = manifest[name]['url']
    ids_before = set(indexed_rag.collection.get(where={'url': url})['ids'])
    indexed_rag.chunk_and_store(
        articles[name]['content'], {'title': articles[name]['title'], 'url': url}
    )
    assert set(indexed_rag.collection.get(where={'url': url})['ids']) == ids_before


def test_retrieval_is_scoped_to_the_current_article(indexed_rag, articles, manifest):
    def foreign_chunks(current, other, question):
        indexed_rag.current_url = manifest[current]['url']
        other_text = normalize(articles[other]['content'])
        return [
            chunk for chunk in indexed_rag.retrieve_relevant_chunks(question)
            if chunk in other_text
        ]

    assert foreign_chunks(
        'varzesh3_2410970', 'mehr_6914285', 'سیره پیامبر چه ویژگی‌هایی داشت؟') == []
    assert foreign_chunks(
        'mehr_6914285', 'varzesh3_2410970', 'سرمربی جدید تراکتور کیست؟') == []


def test_relevant_questions_return_chunks(indexed_rag, manifest):
    for name, questions in IN_ARTICLE.items():
        indexed_rag.current_url = manifest[name]['url']
        for question in questions:
            assert indexed_rag.retrieve_relevant_chunks(question), question


MAX_UNRELATED_WITH_CONTEXT = 1


def test_unrelated_questions_mostly_return_nothing(indexed_rag, manifest):
    admitted = [
        (name, question)
        for name in RETRIEVAL_FIXTURES
        for question in UNRELATED
        if _with_current_url(indexed_rag, manifest, name).retrieve_relevant_chunks(question)
    ]
    assert len(admitted) <= MAX_UNRELATED_WITH_CONTEXT, admitted


def _with_current_url(rag, manifest, name):
    rag.current_url = manifest[name]['url']
    return rag


def test_the_lexical_channel_admits_no_unrelated_question(indexed_rag, manifest):
    for name in RETRIEVAL_FIXTURES:
        indexed_rag.current_url = manifest[name]['url']
        for question in UNRELATED:
            indexed_rag.retrieve(question)
            from_lexical = [
                c for c in indexed_rag.last_retrieval
                if c.kept and c.source in ('lexical', 'both')
            ]
            assert from_lexical == [], (name, question, from_lexical)


def test_ask_returns_the_no_context_message_without_calling_the_model(indexed_rag, manifest):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    assert indexed_rag.ask('پایتخت ژاپن کجاست؟', STRICT) == NO_CONTEXT_MESSAGE


def test_all_returned_chunks_are_within_the_threshold(indexed_rag, manifest):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    question = 'سرمربی جدید تراکتور کیست؟'
    chunks = indexed_rag.retrieve_relevant_chunks(question)
    embedding = indexed_rag.embedding_model.encode([question])[0]
    result = indexed_rag.collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=3,
        where={'url': manifest['varzesh3_2410970']['url']},
        include=['documents', 'distances'],
    )
    kept = {
        doc: dist
        for doc, dist in zip(result['documents'][0], result['distances'][0])
        if doc in chunks
    }
    assert kept
    assert all(dist <= RELEVANCE_THRESHOLD for dist in kept.values())


MIN_GOLD_RECALL = 22
MAX_UNRELATED_ADMITTED = 1


def _gate(rag, manifest, fixture, question):
    rag.current_url = manifest[fixture]['url']
    return rag.retrieve(question)


def test_every_answer_key_is_present_in_its_article(articles, evalset):
    missing = [
        (row['fixture'], row['answer_key'])
        for row in evalset['answerable']
        if row['answer_key'] not in articles[row['fixture']]['content']
    ]
    assert not missing


def test_gold_chunk_recall_meets_the_measured_budget(indexed_rag, manifest, evalset):
    hits = [
        row
        for row in evalset['answerable']
        if any(
            r.kept and row['answer_key'] in r.document
            for r in _gate(indexed_rag, manifest, row['fixture'], row['question'])
        )
    ]
    assert len(hits) >= MIN_GOLD_RECALL, (
        f"{len(hits)} از {len(evalset['answerable'])} — افت نسبت به بودجهٔ سنجیده‌شده"
    )


def test_unrelated_questions_stay_within_the_false_admit_budget(
    indexed_rag, manifest, evalset
):
    admitted = [
        (name, question)
        for question in evalset['unrelated']
        for name in evalset_fixtures(evalset)
        if any(r.kept for r in _gate(indexed_rag, manifest, name, question))
    ]
    assert len(admitted) <= MAX_UNRELATED_ADMITTED, admitted


def evalset_fixtures(evalset):
    seen = []
    for row in evalset['answerable']:
        if row['fixture'] not in seen:
            seen.append(row['fixture'])
    return seen


def test_equivalent_spellings_retrieve_identically(indexed_rag, manifest, evalset):
    for row in evalset['equivalent_spellings']:
        indexed_rag.current_url = manifest[row['fixture']]['url']
        expected = indexed_rag.retrieve_relevant_chunks(row['question'])
        assert expected, row['question']
        assert indexed_rag.retrieve_relevant_chunks(row['variant']) == expected, row


def test_genuine_misspellings_are_not_silently_merged(indexed_rag, manifest, evalset):
    for row in evalset['distinct_spellings']:
        indexed_rag.current_url = manifest[row['fixture']]['url']
        assert indexed_rag.retrieve_relevant_chunks(row['question']) != \
            indexed_rag.retrieve_relevant_chunks(row['variant']), row['why']


def test_query_normalization_is_symmetric_with_the_index(indexed_rag, manifest):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']

    persian = 'سرمربی جدید تراکتور کیست؟'
    arabic_keyboard = 'سرمربي جديد تراكتور كيست؟'
    assert persian != arabic_keyboard

    expected = indexed_rag.retrieve_relevant_chunks(persian)
    assert expected, 'سوال مرجع باید چانک برگرداند'
    assert indexed_rag.retrieve_relevant_chunks(arabic_keyboard) == expected


def test_persian_digits_in_a_question_match_latin_digits_in_the_text(indexed_rag, manifest):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']

    latin = 'تراکتور تقریبا 50 دیدار در سه جام دارد؟'
    persian_digits = 'تراکتور تقریبا ۵۰ دیدار در سه جام دارد؟'

    expected = indexed_rag.retrieve_relevant_chunks(latin)
    assert expected, 'سوال مرجع باید چانک برگرداند'
    assert indexed_rag.retrieve_relevant_chunks(persian_digits) == expected


def test_searching_all_articles_is_possible_with_url_none(indexed_rag, articles, manifest):
    question = 'سیره پیامبر چه ویژگی‌هایی داشت؟'
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    mehr_text = normalize(articles['mehr_6914285']['content'])

    scoped = indexed_rag.retrieve_relevant_chunks(question)
    assert [chunk for chunk in scoped if chunk in mehr_text] == []

    everywhere = indexed_rag.retrieve_relevant_chunks(question, url=None)
    assert [chunk for chunk in everywhere if chunk in mehr_text]


def test_retrieve_reports_blocked_candidates_not_just_kept_ones(indexed_rag, manifest):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    question = 'پایتخت ژاپن کجاست؟'

    assert indexed_rag.retrieve_relevant_chunks(question) == []

    candidates = indexed_rag.retrieve(question)
    assert candidates, 'نامزدها باید برگردند، حتی وقتی همه رد می‌شوند'
    assert not any(c.kept for c in candidates)
    assert min(c.distance for c in candidates) > indexed_rag.relevance_threshold


def test_kept_flag_agrees_with_the_public_chunk_list(indexed_rag, manifest):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    question = 'سرمربی جدید تراکتور کیست؟'

    candidates = indexed_rag.retrieve(question)
    assert [c.document for c in candidates if c.kept] == \
        indexed_rag.retrieve_relevant_chunks(question)
    assert all(
        c.kept == (c.distance <= indexed_rag.relevance_threshold) for c in candidates
    )


def test_candidates_come_back_sorted_by_distance(indexed_rag, manifest):
    indexed_rag.current_url = manifest['mehr_6914285']['url']
    distances = [c.distance for c in indexed_rag.retrieve('سیره پیامبر چیست؟', pool=5)]
    assert distances == sorted(distances)


def test_last_retrieval_records_what_ask_saw(indexed_rag, manifest):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    assert indexed_rag.ask('پایتخت ژاپن کجاست؟', STRICT) == NO_CONTEXT_MESSAGE
    assert indexed_rag.last_retrieval
    assert not any(c.kept for c in indexed_rag.last_retrieval)


def test_threshold_is_per_instance_not_a_module_global(rag_factory, articles, manifest):
    question = 'بودجه باشگاه چقدر است؟'

    strict = rag_factory(relevance_threshold=0.60)
    lenient = rag_factory(relevance_threshold=0.70)

    assert strict.retrieve_relevant_chunks(question) == []
    assert lenient.retrieve_relevant_chunks(question)
    assert strict.retrieve_relevant_chunks(question) == []


def test_the_candidate_pool_is_configurable_per_instance(rag_factory):
    def dense_count(rag):
        return len([c for c in rag.retrieve('تراکتور') if c.source in ('dense', 'both')])

    assert dense_count(rag_factory(candidate_pool=1, prompt_budget=99)) == 1
    assert dense_count(rag_factory(candidate_pool=5, prompt_budget=99)) == 5


def test_the_lexical_channel_adds_beyond_the_candidate_pool(indexed_rag, manifest):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    candidates = indexed_rag.retrieve('تراکتور', pool=3, budget=99)

    assert len([c for c in candidates if c.source in ('dense', 'both')]) == 3
    assert [c for c in candidates if c.source == 'lexical']


def test_lexical_additions_are_capped(indexed_rag, manifest):
    from news_qa.rag import MAX_LEXICAL_ADDITIONS

    indexed_rag.current_url = manifest['mehr_6914285']['url']
    for question in ['پیامبر', 'سیره پیامبر', 'صداقت']:
        additions = [c for c in indexed_rag.retrieve(question) if c.source == 'lexical']
        assert len(additions) <= MAX_LEXICAL_ADDITIONS, question


def test_lexical_candidates_never_duplicate_dense_ones(indexed_rag, manifest):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    for question in ['تراکتور', 'نکونام', 'سرمربی جدید تراکتور کیست؟']:
        documents = [c.document for c in indexed_rag.retrieve(question)]
        assert len(documents) == len(set(documents)), question


def test_lexical_candidates_carry_a_real_distance(indexed_rag, manifest):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    additions = [c for c in indexed_rag.retrieve('تراکتور') if c.source == 'lexical']
    assert additions
    for candidate in additions:
        assert 0.0 <= candidate.distance <= 2.0
        assert candidate.lexical_score > 0


def test_the_headline_is_retrievable(indexed_rag, articles, manifest):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    title = articles['varzesh3_2410970']['title']
    documents = [c.document for c in indexed_rag.retrieve(title)]
    assert normalize(title) in documents


def test_reingesting_rebuilds_the_lexical_index(indexed_rag):
    url = 'https://example.invalid/cache-test'

    indexed_rag.chunk_and_store('نکونام سرمربی تراکتور شد.', {'url': url, 'title': ''})
    indexed_rag.current_url = url
    indexed_rag.retrieve('تراکتور')
    assert url in indexed_rag._lexical_cache

    indexed_rag.chunk_and_store('تراکتور تنها یک جمله دارد.', {'url': url, 'title': ''})
    assert indexed_rag._lexical_cache == {}

    documents, _, _ = indexed_rag._lexical_index(url)
    assert documents == ['تراکتور تنها یک جمله دارد.']

    indexed_rag.collection.delete(where={'url': url})
    indexed_rag._lexical_cache.clear()


def test_the_lexical_channel_rescues_a_gated_dense_candidate(indexed_rag, manifest):
    indexed_rag.current_url = manifest['yjc_9118143']['url']
    candidates = indexed_rag.retrieve('حادثه مدرسه در کدام شهر رخ داد؟')

    rescued = [c for c in candidates if 'میناب' in c.document]
    assert rescued, 'چانکِ حاوی پاسخ باید بین نامزدها باشد'
    assert rescued[0].kept, 'کانال واژگانی باید نجاتش بدهد'
    assert rescued[0].source == 'both', 'باید نشان دهد هر دو کانال دیده‌اندش'
    assert rescued[0].distance > indexed_rag.relevance_threshold, (
        'نکتهٔ آزمون همین است: از آستانهٔ برداری رد نشده و با این حال نگه داشته شده'
    )


def test_a_rescued_candidate_is_not_duplicated(indexed_rag, manifest):
    indexed_rag.current_url = manifest['yjc_9118143']['url']
    documents = [c.document for c in indexed_rag.retrieve('حادثه مدرسه در کدام شهر رخ داد؟')]
    assert len(documents) == len(set(documents))


def _capture_chat(monkeypatch):
    prompts = []

    class _Message:
        content = 'پاسخ ساختگی'
        thinking = ''

    class _Response:
        message = _Message()
        done_reason = 'stop'

    def fake_chat(model, messages, options=None):
        prompts.append(messages[0]['content'])
        return _Response()

    monkeypatch.setattr(news_qa.rag.ollama, 'chat', fake_chat)
    return prompts


def test_strict_mode_never_reaches_the_model_without_context(indexed_rag, manifest, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError('حالت سخت‌گیرانه نباید بدون متن به مدل برسد')

    monkeypatch.setattr(news_qa.rag.ollama, 'chat', explode)
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    assert indexed_rag.ask('پایتخت ژاپن کجاست؟', STRICT) == NO_CONTEXT_MESSAGE


def test_open_mode_does_reach_the_model_without_context(indexed_rag, manifest, monkeypatch):
    prompts = _capture_chat(monkeypatch)
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']

    answer = indexed_rag.ask('پایتخت ژاپن کجاست؟', OPEN)

    assert answer != NO_CONTEXT_MESSAGE
    assert len(prompts) == 1, 'مدل باید دقیقاً یک بار صدا زده شود'
    assert 'پایتخت ژاپن کجاست؟' in prompts[0]
    assert GROUNDING_CLAUSE not in prompts[0]


def test_the_strict_prompt_carries_only_retrieved_chunks(indexed_rag, manifest, monkeypatch):
    prompts = _capture_chat(monkeypatch)
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']

    question = 'سرمربی جدید تراکتور کیست؟'
    chunks = indexed_rag.retrieve_relevant_chunks(question)
    assert chunks, 'سوال مرجع باید متن بگیرد'
    indexed_rag.ask(question, STRICT)

    assert len(prompts) == 1
    for chunk in chunks:
        assert chunk in prompts[0]
    assert GROUNDING_CLAUSE in prompts[0]


def test_the_policy_does_not_change_retrieval(indexed_rag, manifest, monkeypatch):
    _capture_chat(monkeypatch)
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    question = 'سرمربی جدید تراکتور کیست؟'

    indexed_rag.ask(question, STRICT)
    strict_seen = [(c.document, c.kept) for c in indexed_rag.last_retrieval]

    indexed_rag.ask(question, OPEN)
    assert [(c.document, c.kept) for c in indexed_rag.last_retrieval] == strict_seen


def test_ask_records_its_policy_and_clears_stale_thinking(indexed_rag, manifest, monkeypatch):
    monkeypatch.setattr(news_qa.rag.ollama, 'chat', lambda **kw: (_ for _ in ()).throw(RuntimeError()))
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']

    indexed_rag.last_thinking = 'استدلالِ سوالِ قبلی'
    assert indexed_rag.ask('پایتخت ژاپن کجاست؟', STRICT) == NO_CONTEXT_MESSAGE

    assert indexed_rag.last_thinking == ''
    assert indexed_rag.last_policy is STRICT


def test_a_dead_model_service_raises_instead_of_returning_a_string(
    indexed_rag, manifest, monkeypatch
):

    def dead(**kwargs):
        raise ConnectionError('connection refused')

    monkeypatch.setattr(news_qa.rag.ollama, 'chat', dead)
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']

    with pytest.raises(news_qa.rag.GenerationError) as excinfo:
        indexed_rag.ask('سرمربی جدید تراکتور کیست؟', STRICT)

    assert 'connection refused' in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ConnectionError)


def test_a_bug_after_the_call_is_not_disguised_as_a_service_failure(
    indexed_rag, manifest, monkeypatch
):

    class _Broken:
        @property
        def message(self):
            raise AttributeError('شکلِ پاسخ عوض شده')

    monkeypatch.setattr(news_qa.rag.ollama, 'chat', lambda **kw: _Broken())
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']

    with pytest.raises(AttributeError):
        indexed_rag.ask('سرمربی جدید تراکتور کیست؟', STRICT)


def test_an_empty_generation_is_not_an_infrastructure_error(
    indexed_rag, manifest, monkeypatch
):

    class _Message:
        content = ''
        thinking = ''

    class _Empty:
        message = _Message()
        done_reason = 'stop'

    monkeypatch.setattr(news_qa.rag.ollama, 'chat', lambda **kw: _Empty())
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']

    answer = indexed_rag.ask('سرمربی جدید تراکتور کیست؟', STRICT)
    assert 'مدل پاسخی تولید نکرد' in answer


def test_the_shipped_budget_does_not_bind_at_the_shipped_pool(indexed_rag, manifest, evalset):
    for row in evalset['answerable']:
        indexed_rag.current_url = manifest[row['fixture']]['url']
        candidates = indexed_rag.retrieve(row['question'])
        kept = [c.document for c in candidates if c.kept]
        in_prompt = [c.document for c in candidates if c.in_prompt]
        assert kept == in_prompt, row['question']


def test_a_tight_budget_trims_the_prompt_without_touching_the_gate(rag_factory):
    loose = rag_factory(prompt_budget=99)
    tight = rag_factory(prompt_budget=1)

    loose_kept = [c.document for c in loose.retrieve('تراکتور') if c.kept]
    tight_candidates = tight.retrieve('تراکتور')

    assert len(loose_kept) > 1, "سوال باید بیش از یک نامزدِ قبول‌شده بدهد وگرنه آزمون بی‌معناست"
    assert [c.document for c in tight_candidates if c.kept] == loose_kept
    assert len([c for c in tight_candidates if c.in_prompt]) == 1


def test_the_budget_cut_takes_from_both_channels_before_exhausting_one(rag_factory):
    rag = rag_factory(prompt_budget=2)
    sources = [c.source for c in rag.retrieve('تراکتور') if c.in_prompt]

    assert len(sources) == 2
    assert 'lexical' in sources
    assert any(s in ('dense', 'both') for s in sources)


def test_a_deeper_pool_does_not_enlarge_the_prompt(rag_factory):
    shallow = rag_factory(candidate_pool=3, prompt_budget=3)
    deep = rag_factory(candidate_pool=8, prompt_budget=3)

    question = 'تراکتور'
    deep_candidates = deep.retrieve(question)

    assert len([c for c in deep_candidates if c.source in ('dense', 'both')]) > len(
        [c for c in shallow.retrieve(question) if c.source in ('dense', 'both')]
    ), "عمقِ ۸ باید نامزدِ برداریِ بیشتری از عمقِ ۳ بیاورد"
    assert len([c for c in deep_candidates if c.in_prompt]) == 3


def test_a_rejected_candidate_is_never_marked_as_reaching_the_model(rag_factory):
    rag = rag_factory(relevance_threshold=0.30, prompt_budget=99)
    candidates = rag.retrieve('تراکتور')

    assert any(not c.kept for c in candidates), "آستانهٔ تنگ باید دست‌کم یک رد داشته باشد"
    assert all(c.kept for c in candidates if c.in_prompt)


DRIFTING_QUESTION = 'مهم‌ترین بخش سیره پیامبر چیست؟'


def test_a_dense_match_with_no_shared_content_word_is_rejected(rag_factory):
    rag = rag_factory(fixture='yjc_9118143')
    candidates = rag.retrieve(DRIFTING_QUESTION)

    assert candidates, "پرس‌وجو باید نامزد برگرداند وگرنه آزمون چیزی نمی‌سنجد"
    assert min(c.distance for c in candidates) < rag.relevance_threshold, (
        "اگر نزدیک‌ترین نامزد بیرونِ آستانه باشد، این آزمون کفِ پوشش را "
        "نمی‌سنجد بلکه آستانه را می‌سنجد"
    )
    assert not any(c.kept for c in candidates)


def test_turning_the_dense_floor_off_restores_the_leak(rag_factory):
    leaking = rag_factory(fixture='yjc_9118143', dense_coverage=-1)
    assert any(c.kept for c in leaking.retrieve(DRIFTING_QUESTION))


def test_the_floor_leaves_a_lexically_traced_match_alone(rag_factory):
    on, off = rag_factory(), rag_factory(dense_coverage=-1)
    assert [c.document for c in on.retrieve('تراکتور') if c.kept] == [
        c.document for c in off.retrieve('تراکتور') if c.kept
    ]


def test_a_question_of_only_stopwords_is_not_silenced_by_the_floor(rag_factory):
    on = rag_factory(relevance_threshold=0.95)
    off = rag_factory(relevance_threshold=0.95, dense_coverage=-1)

    kept = [c.document for c in on.retrieve('چیست؟') if c.kept]
    assert kept, "قاعده نباید سوالِ بدونِ واژهٔ محتوایی را کاملاً ساکت کند"
    assert kept == [c.document for c in off.retrieve('چیست؟') if c.kept]


def test_the_floor_never_overrules_the_lexical_channel(rag_factory):
    for fixture in ['varzesh3_2410970', 'mehr_6914285', 'yjc_9118143']:
        rag = rag_factory(fixture=fixture)
        for question in ['تراکتور', DRIFTING_QUESTION, 'خبرنگار این گفتگو چه کسی بود؟']:
            candidates = rag.retrieve(question)
            assert all(c.kept for c in candidates if c.source != 'dense'), (
                f"کف چانکی را رد کرد که کانال واژگانی تأییدش کرده بود "
                f"({fixture} / {question})"
            )
