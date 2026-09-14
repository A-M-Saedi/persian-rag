
import math

import pytest

from news_qa.lexical import BM25, STOPWORDS, tokenize


def test_punctuation_does_not_stick_to_words():
    assert tokenize('سرمربی کیست؟') == ['سرمربی']
    assert tokenize('آمد، رفت؛ ماند.') == ['آمد', 'رفت', 'ماند']


def test_half_space_stays_inside_the_word():
    assert tokenize('او می‌کند') == ['می‌کند']


def test_tokenizer_shares_normalization_with_the_index():
    assert tokenize('سرمربي جديد تراكتور كيست؟') == tokenize('سرمربی جدید تراکتور کیست؟')


def test_digits_are_folded_like_the_rest_of_the_pipeline():
    assert tokenize('۱۴۰۴') == tokenize('1404')


def test_stopwords_and_single_characters_are_dropped():
    assert tokenize('این از آن و ب') == []
    assert 'که' in STOPWORDS


DOCS = [
    'نکونام سرمربی تراکتور شد',
    'تراکتور در تبریز بازی می‌کند',
    'حسنلو به تراکتور پیوست',
]


def test_a_rare_term_outranks_a_common_one():
    scores = BM25(DOCS).scores('حسنلو تراکتور')
    assert scores[2] == max(scores)


def test_idf_is_never_negative():
    bm25 = BM25(DOCS)
    assert bm25.idf['تراکتور'] > 0
    assert all(value > 0 for value in bm25.idf.values())


def test_an_empty_corpus_does_not_divide_by_zero():
    assert BM25([]).scores('تراکتور') == []
    assert BM25(['', '  ']).scores('تراکتور') == [0.0, 0.0]


def test_a_query_with_no_known_terms_scores_zero():
    assert BM25(DOCS).scores('قرمه‌سبزی') == [0.0, 0.0, 0.0]


def test_coverage_is_one_when_every_query_term_is_present():
    assert BM25(DOCS).coverage('نکونام سرمربی')[0] == pytest.approx(1.0)


def test_absent_query_terms_pull_coverage_down():
    bm25 = BM25(DOCS)
    both = bm25.coverage('تراکتور تبریز')[1]
    one_of_three = bm25.coverage('تراکتور قرمه‌سبزی برنامه‌نویسی')[1]
    assert one_of_three < both
    assert one_of_three < 0.35, 'باید زیر کفِ عملیاتی بیفتد، وگرنه بی‌اثر است'


def test_a_completely_foreign_query_covers_nothing():
    assert BM25(DOCS).coverage('قرمه‌سبزی برنامه‌نویسی') == [0.0, 0.0, 0.0]


def test_coverage_of_an_empty_query_is_zero_not_a_crash():
    assert BM25(DOCS).coverage('این از آن') == [0.0, 0.0, 0.0]


def test_coverage_stays_within_the_unit_interval():
    bm25 = BM25(DOCS)
    for query in ['تراکتور', 'نکونام سرمربی تبریز', 'حسنلو قرمه‌سبزی']:
        assert all(0.0 <= value <= 1.0 for value in bm25.coverage(query)), query


def test_absent_terms_weigh_at_least_as_much_as_the_rarest_present_one():
    bm25 = BM25(DOCS)
    assert bm25.max_idf >= max(bm25.idf.values())
    assert bm25._weight('واژه‌ای‌که‌نیست') == bm25.max_idf
