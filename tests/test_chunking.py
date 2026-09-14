import re


def _token_count(rag, text):
    return len(rag.embedding_model.tokenizer.encode(text))


def test_no_chunk_exceeds_the_embedding_token_limit(rag, articles, fixture_name):
    limit = rag.embedding_model.max_seq_length
    chunks = rag.text_splitter.split_text(articles[fixture_name]['content'])
    oversized = [c for c in chunks if _token_count(rag, c) > limit]
    assert not oversized, f'{len(oversized)} از {len(chunks)} چانک از سقف {limit} توکن رد شد'


def test_chunks_are_verbatim_substrings_of_the_source(rag, articles, fixture_name):
    text = articles[fixture_name]['content']
    for chunk in rag.text_splitter.split_text(text):
        assert chunk in text


def test_chunks_do_not_start_or_end_mid_word(rag, articles, fixture_name):
    text = articles[fixture_name]['content']
    for chunk in rag.text_splitter.split_text(text):
        start = text.index(chunk)
        end = start + len(chunk)
        if start > 0:
            assert text[start - 1].isspace(), f'چانک وسط واژه شروع می‌شود: {chunk[:40]!r}'
        if end < len(text):
            assert text[end].isspace(), f'چانک وسط واژه تمام می‌شود: {chunk[-40:]!r}'


def test_paragraph_separator_is_preferred(rag):
    para = 'این یک جملهٔ نمونه برای آزمون چانک‌بندی فارسی است و به اندازهٔ کافی بلند است. ' * 4
    text = f'{para}\n\n{para}'
    chunks = rag.text_splitter.split_text(text)
    assert len(chunks) > 1
    assert all('\n\n' not in c for c in chunks)


def test_chunk_boundaries_land_on_sentence_ends_more_often_than_not(rag, articles):
    chunks = rag.text_splitter.split_text(articles['mehr_6914285']['content'])
    enders = ('.', '؟', '!', '؛', ':', '،')
    ending_well = sum(1 for c in chunks if c.rstrip().endswith(enders))
    assert ending_well / len(chunks) >= 0.5


def test_no_chunk_is_empty_or_whitespace_only(rag, articles, fixture_name):
    for chunk in rag.text_splitter.split_text(articles[fixture_name]['content']):
        assert chunk.strip()


def test_every_word_of_the_source_survives_chunking(rag, articles, fixture_name):
    text = articles[fixture_name]['content']
    chunks = rag.text_splitter.split_text(text)
    covered = set()
    for chunk in chunks:
        covered.update(chunk.split())
    missing = [w for w in text.split() if w not in covered]
    assert not missing, f'{len(missing)} واژه گم شد، نمونه: {missing[:5]}'
