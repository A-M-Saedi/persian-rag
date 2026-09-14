import copy

import pytest

from news_qa import sources
from news_qa.document import KIND_PDF, KIND_TEXT, KIND_WEB, Document, SourceError
from news_qa.scraper import THIN_ARTICLE_WORDS, ScrapeError
from news_qa.sources import files, layout, pdf
from news_qa.sources.layout import TextLine


def test_a_network_failure_is_still_a_scrape_error_and_also_a_source_error():
    assert issubclass(ScrapeError, SourceError)
    assert not issubclass(SourceError, ScrapeError)


def test_a_document_reports_its_own_word_count():
    assert Document('ت', 'یک دو سه', 'x', KIND_WEB).word_count == 3


def test_an_unknown_suffix_names_the_formats_that_do_work(tmp_path):
    path = tmp_path / 'گزارش.docx'
    path.write_text('x', encoding='utf-8')

    with pytest.raises(SourceError) as error:
        files.load_file(path)

    assert '.pdf' in str(error.value) and '.txt' in str(error.value)


def test_a_missing_file_is_a_source_error_not_a_traceback(tmp_path):
    with pytest.raises(SourceError):
        files.load_file(tmp_path / 'نیست.txt')


def test_a_directory_is_rejected(tmp_path):
    with pytest.raises(SourceError):
        files.load_file(tmp_path)


def test_every_registered_suffix_is_lowercase_and_dotted():
    for suffix in files.EXTRACTORS:
        assert suffix.startswith('.') and suffix == suffix.lower()


def test_an_uppercase_suffix_still_resolves(tmp_path):
    path = tmp_path / 'یادداشت.TXT'
    path.write_text('سلام دنیا', encoding='utf-8')

    assert files.load_file(path).kind == KIND_TEXT


def test_a_local_file_gets_a_file_uri_as_its_source_id(tmp_path):
    path = tmp_path / 'سند.txt'
    path.write_text('متن', encoding='utf-8')

    assert files.load_file(path).source_id.startswith('file://')


def test_the_same_file_reached_by_two_paths_gets_one_identity(tmp_path):
    path = tmp_path / 'سند.txt'
    path.write_text('متن', encoding='utf-8')
    indirect = tmp_path / 'زیر' / '..' / 'سند.txt'
    (tmp_path / 'زیر').mkdir()

    assert files.load_file(path).source_id == files.load_file(indirect).source_id


def test_plain_text_keeps_paragraph_boundaries(tmp_path):
    path = tmp_path / 'یادداشت.txt'
    path.write_text('پاراگراف\nیک\n\nپاراگراف دو', encoding='utf-8')

    assert files.load_file(path).content == 'پاراگراف\nیک\n\nپاراگراف دو'


def test_the_file_name_becomes_a_searchable_title(tmp_path):
    path = tmp_path / 'گزارش_سالانه-۱۴۰۲.txt'
    path.write_text('متن', encoding='utf-8')

    assert files.load_file(path).title == 'گزارش سالانه ۱۴۰۲'


def test_a_non_utf8_file_fails_loudly_instead_of_producing_garbage(tmp_path):
    path = tmp_path / 'سند.txt'
    path.write_bytes('سلام'.encode('windows-1256'))

    with pytest.raises(SourceError) as error:
        files.load_file(path)

    assert 'UTF-8' in str(error.value)


def test_the_thin_floor_for_files_is_not_the_calibrated_web_number():
    assert sources.thin_threshold(KIND_WEB) == THIN_ARTICLE_WORDS
    assert sources.thin_threshold(KIND_PDF) < THIN_ARTICLE_WORDS


def test_a_short_but_healthy_file_is_not_flagged_as_thin():
    document = Document('صورت‌جلسه', 'واژه ' * 80, 'file:///x.pdf', KIND_PDF)

    assert not sources.is_thin(document)


def test_a_nearly_empty_extraction_is_flagged_whatever_the_kind():
    assert sources.is_thin(Document('ت', 'سه واژه دارد', 'file:///x.pdf', KIND_PDF))


def _raw_page(lines):
    return {'blocks': [{'type': 0, 'lines': lines}]}


def _raw_line(spans, bbox):
    return {'bbox': bbox, 'spans': spans}


def _span(text, x0, x1, size=10.0, flags=0, font='Nazanin'):
    return {'text': text, 'bbox': [x0, 0, x1, 12], 'size': size, 'flags': flags, 'font': font}


def test_image_blocks_are_skipped_because_they_hold_no_text():
    raw = {'blocks': [
        {'type': 1, 'bbox': [0, 0, 10, 10]},
        {'type': 0, 'lines': [_raw_line([_span('متن', 50, 100)], [50, 0, 100, 12])]},
    ]}

    assert [item.text for item in pdf.page_lines(raw, rtl=True)] == ['متن']


def test_a_line_takes_the_largest_span_size_not_the_average():
    raw = _raw_page([_raw_line(
        [_span('عنوان', 50, 150, size=18.0), _span('۱', 150, 155, size=7.0)],
        [50, 0, 155, 20],
    )])

    assert pdf.page_lines(raw, rtl=False)[0].size == 18.0


def test_a_line_counts_as_bold_only_when_every_span_is_bold():
    partly = _raw_page([_raw_line(
        [_span('متن', 50, 90, flags=1 << 4), _span('عادی', 90, 140)],
        [50, 0, 140, 12],
    )])
    fully = _raw_page([_raw_line(
        [_span('عنوان', 50, 140, font='IRANSans-Bold')], [50, 0, 140, 12],
    )])

    assert pdf.page_lines(partly, rtl=False)[0].bold is False
    assert pdf.page_lines(fully, rtl=False)[0].bold is True


def test_lines_inside_a_table_are_taken_out_of_the_prose_flow():
    above = TextLine('بالای جدول', 50, 100, 300, 112)
    inside = TextLine('خانهٔ جدول', 50, 210, 300, 222)
    below = TextLine('زیر جدول', 50, 320, 300, 332)
    bands = [(200.0, 300.0, [['a', 'b']])]

    groups = pdf._split_by_bands([above, inside, below], bands)

    assert [item.text for group in groups for item in group[1]] == ['بالای جدول', 'زیر جدول']


def test_with_no_tables_every_line_stays_in_one_group():
    lines = [TextLine('یک', 0, 0, 10, 10), TextLine('دو', 0, 20, 10, 30)]

    assert pdf._split_by_bands(lines, []) == [(None, lines)]


def test_junk_metadata_titles_are_rejected():
    assert pdf._title_from_metadata({'title': 'Microsoft Word - doc1.doc'}) == ''
    assert pdf._title_from_metadata({'title': 'گزارش سالانه'}) == 'گزارش سالانه'


def test_the_title_falls_back_to_the_largest_line_on_the_first_page():
    lines = [
        TextLine('گزارش وضعیت آب', 50, 40, 300, 62, size=18),
        TextLine('متن عادی صفحه', 50, 90, 300, 102, size=10),
        TextLine('باز هم متن عادی', 50, 110, 300, 122, size=10),
    ]

    assert pdf.title_from_first_page(lines) == 'گزارش وضعیت آب'


def test_a_single_size_document_has_no_detectable_title():
    lines = [TextLine(f'سطر {i}', 50, 40 + i * 20, 300, 52 + i * 20, size=10) for i in range(5)]

    assert pdf.title_from_first_page(lines) == ''


def test_a_mostly_imageless_pdf_reports_scanning_instead_of_failing():
    notes = pdf._notes(page_count=10, empty_pages=list(range(1, 9)), tables=0, words=30)

    assert any('OCR' in note for note in notes)


def test_a_few_blank_pages_are_listed_rather_than_called_a_scan():
    notes = pdf._notes(page_count=20, empty_pages=[3, 7], tables=0, words=4000)

    assert notes and 'OCR' not in notes[0] and '3' in notes[0]


def test_a_clean_document_produces_no_notes():
    assert pdf._notes(page_count=5, empty_pages=[], tables=0, words=2000) == ()


def test_pymupdf_is_only_imported_when_a_pdf_is_actually_read(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'fitz', None)
    monkeypatch.setattr(
        pdf, '_require_fitz',
        lambda: (_ for _ in ()).throw(SourceError('pip install pymupdf')),
    )

    with pytest.raises(SourceError) as error:
        pdf.load(__import__('pathlib').Path('x.pdf'))

    assert 'pymupdf' in str(error.value)


def test_a_file_document_flows_through_the_unchanged_rag_pipeline(rag, tmp_path):
    path = tmp_path / 'باشگاه.txt'
    path.write_text(
        'باشگاه تراکتور در فصل جاری عملکرد خوبی داشته است.\n\n'
        'سرمربی تیم از هواداران برای حمایتشان تشکر کرد.',
        encoding='utf-8',
    )
    document = files.load_file(path)

    clone = copy.copy(rag)
    clone.collection = rag.chroma_client.get_or_create_collection(
        name='file_source_test', metadata={"hnsw:space": "cosine"}
    )
    clone._lexical_cache = {}
    clone.chunk_and_store(document.content, {
        'title': document.title, 'url': document.source_id, 'kind': document.kind,
    })

    found = clone.retrieve('سرمربی چه گفت؟', url=document.source_id)

    assert found, 'سندِ فایل از راهِ بازیابیِ معمول پیدا نشد'
    assert any('سرمربی' in item.document for item in found)
    assert clone.retrieve('سرمربی چه گفت؟', url='https://example.com/other') == []


def test_re_ingesting_the_same_file_replaces_rather_than_duplicates(rag, tmp_path):
    path = tmp_path / 'سند.txt'
    path.write_text('متن نمونه برای آزمون تکرار.', encoding='utf-8')
    document = files.load_file(path)

    clone = copy.copy(rag)
    clone.collection = rag.chroma_client.get_or_create_collection(
        name='file_reingest_test', metadata={"hnsw:space": "cosine"}
    )
    clone._lexical_cache = {}
    clone.chunk_and_store(document.content, {'title': document.title, 'url': document.source_id})
    first = clone.collection.count()
    clone.chunk_and_store(document.content, {'title': document.title, 'url': document.source_id})

    assert clone.collection.count() == first


def test_a_persian_digit_run_glued_to_a_word_is_separated():
    assert layout.split_bidi_runs('صدور۱۴۰۴/۳/۴') == 'صدور ۱۴۰۴/۳/۴'


def test_a_latin_digit_run_glued_to_a_word_is_separated():
    assert layout.split_bidi_runs('تخلیه و2,000,000 ریال') == 'تخلیه و 2,000,000 ریال'


def test_text_that_is_already_spaced_is_left_alone():
    for text in ('از ۱۴۰۴ تا ۱۴۰۵', 'یک متن فارسی بدون عدد', 'plain english 2024'):
        assert layout.split_bidi_runs(text) == text


def test_a_ceremonial_heading_is_not_used_as_the_title():
    lines = [
        layout.TextLine('بسمه تعالی', 250, 40, 345, 60, 18),
        layout.TextLine('اخطاریه تخلیه و سلب امتیاز', 150, 80, 445, 100, 16),
    ] + [
        layout.TextLine('متن عادی سند که ادامه دارد', 60, 130 + i * 14,
                        400, 142 + i * 14, 10)
        for i in range(6)
    ]
    assert pdf.title_from_first_page(lines) == 'اخطاریه تخلیه و سلب امتیاز'


def test_quotes_around_a_title_are_stripped():
    lines = [
        layout.TextLine('"اخطاریه تخلیه"', 150, 80, 445, 100, 16),
    ] + [
        layout.TextLine('متن عادی سند که ادامه دارد', 60, 130 + i * 14,
                        400, 142 + i * 14, 10)
        for i in range(6)
    ]
    assert pdf.title_from_first_page(lines) == 'اخطاریه تخلیه'
