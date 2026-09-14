import hashlib

import pytest
import requests

from news_qa.scraper import DEFAULT_TITLE, ScrapeError

ZWNJ = '‌'


def test_clean_text_delegates_to_the_shared_normalizer(scraper):
    from news_qa.normalize import normalize

    samples = ['كيف ۱۲۳', f'می{ZWNJ}شود', 'مسئولیت', 'اول\n\n\n\nدوم']
    assert [scraper.clean_text(s) for s in samples] == [normalize(s) for s in samples]


def test_inline_tags_do_not_glue_neighbouring_words(scraper):
    html = (
        '<article><p>تیم فوتبال<a href="/x">پرسپولیس</a>در اولین بازی فصل '
        'با <a>مهدی تارتار</a>بدون مصدوم به میدان می‌رود.</p></article>'
    )
    content = scraper.extract_from_html(html)['content']
    assert 'فوتبالپرسپولیس' not in content
    assert 'تارتاربدون' not in content
    assert 'تیم فوتبال پرسپولیس در' in content
    assert 'مهدی تارتار بدون' in content


def test_paragraphs_shorter_than_the_minimum_are_dropped(scraper):
    html = '<article><p>کوتاه</p><p>' + 'متن به اندازه کافی بلند برای ماندن ' * 2 + '</p></article>'
    content = scraper.extract_from_html(html)['content']
    assert 'کوتاه' not in content
    assert 'به اندازه کافی بلند' in content


def test_noise_tags_are_removed(scraper):
    html = (
        '<html><body><script>var x = 1;</script>'
        '<footer><p>تماس با ما و اطلاعات تکمیلی سایت اینجاست</p></footer>'
        '<article><p>این متن اصلی خبر است و باید در خروجی بماند حتما</p></article>'
        '</body></html>'
    )
    content = scraper.extract_from_html(html)['content']
    assert 'متن اصلی خبر' in content
    assert 'تماس با ما' not in content
    assert 'var x' not in content


def test_falls_back_to_all_paragraphs_when_no_selector_matches(scraper):
    html = '<html><body><div><p>' + 'متنی که در هیچ انتخابگر شناخته‌شده‌ای نیست ' * 2 + '</p></div></body></html>'
    assert 'هیچ انتخابگر' in scraper.extract_from_html(html)['content']


def test_missing_title_falls_back(scraper):
    assert scraper.extract_from_html('<html><body><p>x</p></body></html>')['title'] == DEFAULT_TITLE


def test_title_with_nested_markup_does_not_crash(scraper):
    html = (
        '<html><head><title>خبر <b>مهم</b> امروز</title></head>'
        '<body><article><p>' + 'متن خبر با طول کافی برای ماندن در خروجی ' * 2 +
        '</p></article></body></html>'
    )
    article = scraper.extract_from_html(html)
    assert article['title'] == 'خبر مهم امروز'
    assert 'متن خبر' in article['content']


def test_empty_title_falls_back(scraper):
    html = '<html><head><title></title></head><body><p>x</p></body></html>'
    assert scraper.extract_from_html(html)['title'] == DEFAULT_TITLE


def test_og_title_wins_over_the_document_title(scraper):
    html = (
        '<html><head><meta property="og:title" content="یک بمب خبری دیگر در تراکتور!">'
        '<title>یک بمب خبری دیگر در تراکتور! | ورزش سه</title></head>'
        '<body><p>x</p></body></html>'
    )
    assert scraper.extract_from_html(html)['title'] == 'یک بمب خبری دیگر در تراکتور!'


def test_empty_og_title_falls_back_to_the_document_title(scraper):
    html = (
        '<html><head><meta property="og:title" content="">'
        '<title>عنوان واقعی</title></head><body><p>x</p></body></html>'
    )
    assert scraper.extract_from_html(html)['title'] == 'عنوان واقعی'


def test_extraction_matches_recorded_baseline(scraper, load_html, baseline, fixture_name):
    article = scraper.extract_from_html(load_html(fixture_name))
    expected = baseline[fixture_name]
    assert article['title'] == expected['title']
    assert len(article['content'].split()) == expected['content_words']
    assert hashlib.sha256(article['content'].encode()).hexdigest() == expected['content_sha256']


def test_network_failure_raises_scrape_error(scraper, monkeypatch):
    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError('network is unreachable')

    monkeypatch.setattr(requests, 'get', boom)
    with pytest.raises(ScrapeError, match='network is unreachable'):
        scraper.extract_article('https://example.com/news/1')


def test_programming_errors_are_not_disguised_as_network_errors(scraper, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError('a bug, not a network problem')

    monkeypatch.setattr(scraper, 'extract_from_html', boom)
    monkeypatch.setattr(scraper, 'fetch', lambda url: '<html></html>')
    with pytest.raises(ValueError):
        scraper.extract_article('https://example.com/news/1')
