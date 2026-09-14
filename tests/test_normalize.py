import pytest

from news_qa.normalize import normalize

ZWNJ = '‌'


def test_arabic_ya_and_kaf_become_persian():
    assert normalize('كيف') == 'کیف'
    assert normalize('سرمربي جديد تراكتور كيست؟') == 'سرمربی جدید تراکتور کیست؟'


def test_persian_and_arabic_digits_become_latin():
    assert normalize('۱۲۳ و ٤٥٦') == '123 و 456'


@pytest.mark.parametrize('word', [
    'مسئولیت', 'مسئله', 'رئیس', 'جزئیات', 'هیئت', 'ارائه', 'قائل', 'مسئولان',
])
def test_hamza_on_ya_is_preserved(word):
    assert normalize(word) == word


@pytest.mark.parametrize('source,expected', [
    ('تأکید', 'تاکید'),
    ('تأسی', 'تاسی'),
    ('مؤثر', 'موثر'),
    ('مکالمة', 'مکالمه'),
])
def test_accepted_spelling_variants_are_merged(source, expected):
    assert normalize(source) == expected


def test_zwnj_is_not_treated_as_whitespace():
    assert normalize(f'می{ZWNJ}شود و می{ZWNJ}رود').count(ZWNJ) == 2


def test_paragraph_breaks_survive():
    assert normalize('اول\n\nدوم') == 'اول\n\nدوم'
    assert normalize('اول\n\n\n\nدوم') == 'اول\n\nدوم'


def test_horizontal_whitespace_is_collapsed():
    assert normalize('یک   \t  دو') == 'یک دو'


@pytest.mark.parametrize('text', [
    'كيف ۱۲۳',
    'مسئولیت مؤثر',
    'اول\n\n\n\nدوم   سوم',
    f'می{ZWNJ}شود',
])
def test_normalize_is_idempotent(text):
    once = normalize(text)
    assert normalize(once) == once
