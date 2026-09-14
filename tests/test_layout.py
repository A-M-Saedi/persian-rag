import pytest

from news_qa.normalize import normalize
from news_qa.sources import layout
from news_qa.sources.layout import BODY, Page, Paragraph, TextLine


def line(text, x0=50, y0=100, width=None, height=12, size=10, bold=False):
    width = width if width is not None else len(text) * 5.0
    return TextLine(text, x0, y0, x0 + width, y0 + height, size, bold)


def rtl_line(text, y0, width, right=340.0, height=12, size=10):
    return TextLine(text, right - width, y0, right, y0 + height, size, False)


def column(texts, x0, y0=100, width=200, step=14, size=10):
    return [
        line(text, x0=x0, y0=y0 + index * step, width=width, size=size)
        for index, text in enumerate(texts)
    ]


def test_arabic_presentation_forms_become_ordinary_letters():
    assert layout.clean_unicode('ﺱﻼﻡ') == 'سلام'
    assert layout.clean_unicode('ﻻ') == 'لا'


def test_zwnj_survives_but_invisible_controls_do_not():
    assert layout.clean_unicode('می‌رود') == 'می‌رود'
    assert layout.clean_unicode('‫متن‬​­') == 'متن'


def test_digits_are_left_for_normalize_to_handle():
    assert layout.clean_unicode('۱۲۳') == '۱۲۳'
    assert normalize(layout.clean_unicode('۱۲۳')) == '123'


def test_spans_of_a_right_to_left_line_are_read_from_the_right():
    spans = [('دنیا', 10.0, 50.0), ('سلام', 60.0, 100.0)]

    assert layout.join_spans(spans, rtl=True) == 'سلام دنیا'
    assert layout.join_spans(spans, rtl=False) == 'دنیا سلام'


def test_adjacent_spans_do_not_get_a_space_between_them():
    joined = layout.join_spans([('inter', 10.0, 40.0), ('national', 40.0, 88.0)], rtl=False)

    assert joined == 'international'


def test_a_real_horizontal_gap_still_produces_a_space():
    joined = layout.join_spans([('سلام', 60.0, 100.0), ('دنیا', 10.0, 45.0)], rtl=True)

    assert joined == 'سلام دنیا'


def _paged_book(pages=6):
    out = []
    for number in range(1, pages + 1):
        out.append(Page(number, 400.0, 800.0, (
            line('فصل دوم: روش‌ها', x0=50, y0=20, width=200),
            line(f'متن اصلی صفحهٔ {number}', x0=50, y0=300, width=300),
            line(f'صفحه {number}', x0=180, y0=760, width=60),
        )))
    return out


def test_running_headers_and_numbered_footers_are_removed():
    text = layout.render(layout.document_paragraphs(_paged_book()))

    assert 'فصل دوم' not in text
    assert 'صفحه 1' not in text and 'صفحه ۱' not in text
    assert text.count('متن اصلی') == 6


def test_a_repeated_line_in_the_body_is_not_mistaken_for_furniture():
    pages = [
        Page(number, 400.0, 800.0, (
            line('شعار سازمان', x0=50, y0=400, width=200),
            line(f'متن {number}', x0=50, y0=430, width=200),
        ))
        for number in range(1, 6)
    ]

    text = layout.render(layout.document_paragraphs(pages))

    assert text.count('شعار سازمان') == 5


def test_furniture_detection_is_skipped_for_very_short_documents():
    assert layout.find_furniture(_paged_book(pages=2)) == frozenset()


def test_a_duplicated_text_layer_is_collapsed():
    duplicated = [line('یک جمله', x0=50, y0=100, width=200)] * 2
    page = Page(1, 400.0, 800.0, tuple(duplicated))

    assert layout.render(layout.document_paragraphs([page])) == 'یک جمله'


def test_two_columns_are_read_right_first_in_a_persian_document():
    left = column(['چهار', 'پنج', 'شش'], x0=40, width=140)
    right = column(['یک', 'دو', 'سه'], x0=220, width=140)
    page = Page(1, 400.0, 800.0, tuple(left + right))

    ordered = [item.text for item in layout.reading_order(page, rtl=True)]

    assert ordered == ['یک', 'دو', 'سه', 'چهار', 'پنج', 'شش']


def test_the_same_page_is_read_left_first_when_the_text_is_latin():
    left = column(['one', 'two', 'three'], x0=40, width=140)
    right = column(['four', 'five', 'six'], x0=220, width=140)
    page = Page(1, 400.0, 800.0, tuple(left + right))

    ordered = [item.text for item in layout.reading_order(page, rtl=False)]

    assert ordered == ['one', 'two', 'three', 'four', 'five', 'six']


def test_a_full_width_heading_does_not_disable_column_detection():
    heading = line('عنوان بخش', x0=40, y0=50, width=320, size=16)
    left = column(['چهار', 'پنج', 'شش'], x0=40, y0=100, width=140)
    right = column(['یک', 'دو', 'سه'], x0=220, y0=100, width=140)
    page = Page(1, 400.0, 800.0, tuple([heading] + left + right))

    ordered = [item.text for item in layout.reading_order(page, rtl=True)]

    assert ordered == ['عنوان بخش', 'یک', 'دو', 'سه', 'چهار', 'پنج', 'شش']


def test_ragged_single_column_text_is_not_split_into_columns():
    lines = [
        line('یک سطر بلند از متن اصلی', x0=40, y0=100, width=320),
        line('کوتاه', x0=40, y0=120, width=60),
        line('یک سطر بلند دیگر از متن', x0=40, y0=140, width=320),
        line('باز هم کوتاه', x0=40, y0=160, width=90),
        line('سطر پایانی متن این بخش', x0=40, y0=180, width=310),
        line('پایان', x0=40, y0=200, width=50),
    ]

    assert layout.find_gutter(lines) is None


def test_a_wide_vertical_gap_starts_a_new_paragraph():
    lines = [
        line('سطر یک از پاراگراف نخست', x0=40, y0=100, width=300),
        line('سطر دو از پاراگراف نخست', x0=40, y0=116, width=300),
        line('پاراگراف دوم از اینجا', x0=40, y0=180, width=300),
    ]

    paragraphs = layout.lines_to_paragraphs(lines, rtl=True)

    assert len(paragraphs) == 2
    assert paragraphs[0].text == 'سطر یک از پاراگراف نخست سطر دو از پاراگراف نخست'


def test_a_short_line_that_does_not_end_a_sentence_keeps_the_paragraph_open():
    lines = [
        rtl_line('این سطر بلند است و ادامه دارد', y0=100, width=300),
        rtl_line('کوتاه ولی ادامه‌دار', y0=116, width=190),
        rtl_line('و پایانِ جمله اینجاست.', y0=132, width=300),
    ]

    assert len(layout.lines_to_paragraphs(lines, rtl=True)) == 1


def test_a_heading_becomes_its_own_paragraph():
    lines = [
        line('مقدمه', x0=40, y0=100, width=80, size=16, bold=True),
        line('متن مقدمه از اینجا آغاز می‌شود', x0=40, y0=126, width=300, size=10),
        line('و در همین سطر ادامه دارد و', x0=40, y0=142, width=300, size=10),
    ]

    paragraphs = layout.lines_to_paragraphs(lines, rtl=True)

    assert [p.kind for p in paragraphs] == [layout.HEADING, BODY]
    assert paragraphs[0].text == 'مقدمه'


def test_list_items_stay_separate_paragraphs():
    lines = [
        line('۱. مورد نخست', x0=40, y0=100, width=200),
        line('۲. مورد دوم', x0=40, y0=116, width=200),
        line('- مورد سوم', x0=40, y0=132, width=200),
    ]

    paragraphs = layout.lines_to_paragraphs(lines, rtl=True)

    assert len(paragraphs) == 3
    assert all(p.kind == layout.LIST for p in paragraphs)


def test_an_english_word_broken_across_lines_is_rejoined():
    assert layout.join_lines(['the inter-', 'national body']) == 'the international body'


def test_a_line_ending_in_zwnj_joins_without_a_space():
    assert layout.join_lines(['می‌', 'رود']) == 'می‌رود'


def test_a_hyphen_between_two_persian_lines_is_left_alone():
    assert layout.join_lines(['تهران -', 'کرج']) == 'تهران - کرج'


def test_a_paragraph_cut_by_a_page_break_is_stitched_back():
    first = [Paragraph('این جمله در انتهای صفحه بریده', BODY)]
    second = [Paragraph('شد و در صفحهٔ بعد ادامه یافت.', BODY)]

    merged = layout.merge_pages([first, second])

    assert len(merged) == 1
    assert merged[0].text == 'این جمله در انتهای صفحه بریده شد و در صفحهٔ بعد ادامه یافت.'


def test_a_completed_sentence_is_not_stitched_to_the_next_page():
    first = [Paragraph('این جمله تمام شد.', BODY)]
    second = [Paragraph('و این یکی تازه شروع می‌شود.', BODY)]

    assert len(layout.merge_pages([first, second])) == 2


def test_a_heading_at_the_top_of_a_page_is_never_stitched():
    first = [Paragraph('جمله‌ای که ناتمام مانده', BODY)]
    second = [Paragraph('فصل سوم', layout.HEADING)]

    assert len(layout.merge_pages([first, second])) == 2


def test_a_table_row_is_rewritten_as_header_value_pairs():
    rows = [['سال', 'شهر'], ['۱۴۰۲', 'تهران']]

    rendered = layout.render_table(rows, rtl=False)

    assert rendered == ['سال | شهر', 'سال: ۱۴۰۲ | شهر: تهران']


def test_table_columns_are_reversed_for_a_right_to_left_document():
    rendered = layout.render_table([['سال', 'شهر'], ['۱۴۰۲', 'تهران']], rtl=True)

    assert rendered[0] == 'شهر | سال'


def test_a_table_whose_first_row_has_a_blank_cell_is_not_treated_as_headed():
    rendered = layout.render_table([['الف', ''], ['ب', 'ج']], rtl=False)

    assert rendered == ['الف', 'ب | ج']


def test_each_table_row_is_a_separate_paragraph():
    rendered = layout.render_table([['a', 'b'], ['1', '2'], ['3', '4']], rtl=False)

    assert len(rendered) == 3


def test_the_output_never_carries_page_markers_or_placeholders():
    pages = [Page(1, 400.0, 800.0, (line('متن صفحهٔ اول', width=200),)),
             Page(2, 400.0, 800.0, ())]

    text = layout.render(layout.document_paragraphs(pages))

    assert 'Page' not in text
    assert '[' not in text
    assert text == 'متن صفحهٔ اول'


def test_paragraphs_are_separated_by_a_blank_line():
    rendered = layout.render([Paragraph('الف', BODY), Paragraph('ب', BODY)])

    assert rendered == 'الف\n\nب'


def test_the_output_survives_normalize_unchanged_in_structure():
    rendered = layout.render([Paragraph('سطر یک', BODY), Paragraph('سطر دو', BODY)])

    assert normalize(rendered).count('\n\n') == 1


def test_a_line_ending_in_a_hyphen_is_not_a_paragraph_end():
    lines = [
        layout.TextLine('a very long line of ordinary body text here', 60, 100, 400, 110, 10),
        layout.TextLine('The distri-', 60, 112, 120, 122, 10),
        layout.TextLine('bution network was audited.', 60, 124, 260, 134, 10),
    ]
    paragraphs = layout.lines_to_paragraphs(lines, rtl=False)
    assert len(paragraphs) == 1
    assert 'distribution network' in paragraphs[0].text
    assert 'distri-' not in paragraphs[0].text


def test_a_line_ending_in_zwnj_is_not_a_paragraph_end():
    lines = [
        layout.TextLine('یک سطر کامل از متن بدنه که تا حاشیه می‌رسد', 60, 100, 400, 110, 10),
        layout.TextLine('می‌', 60, 112, 90, 122, 10),
        layout.TextLine('رود تا پایان دوره', 60, 124, 200, 134, 10),
    ]
    paragraphs = layout.lines_to_paragraphs(lines, rtl=False)
    assert len(paragraphs) == 1
    assert 'می‌رود' in paragraphs[0].text


def test_a_short_line_without_a_hyphen_still_ends_the_paragraph():
    lines = [
        layout.TextLine('a very long line of ordinary body text here', 60, 100, 400, 110, 10),
        layout.TextLine('short.', 60, 112, 100, 122, 10),
        layout.TextLine('A new paragraph begins on this line here.', 60, 124, 380, 134, 10),
    ]
    assert len(layout.lines_to_paragraphs(lines, rtl=False)) > 1


def test_a_dash_after_a_digit_is_not_treated_as_a_broken_word():
    assert layout.word_continues('inter-')
    assert not layout.word_continues('1401-')
    assert not layout.word_continues('a normal line.')


def test_a_short_latin_line_in_an_rtl_document_is_not_read_as_an_indent():
    lines = [
        layout.TextLine('یک سطر بلند فارسی که تا لبهٔ راست متن می‌رسد', 60, 100, 470, 114, 10),
        layout.TextLine('The distri-', 60, 118, 104, 132, 10),
        layout.TextLine('bution grid was audited.', 60, 136, 184, 150, 10),
    ]
    paragraphs = layout.lines_to_paragraphs(lines, rtl=True)
    assert 'distribution grid was audited.' in ' '.join(p.text for p in paragraphs)


def test_a_line_crossing_the_gutter_does_not_cancel_column_detection():
    lines = []
    for i in range(6):
        lines.append(layout.TextLine('R%d' % i, 330, 140 + i * 20, 490, 152 + i * 20, 10))
        lines.append(layout.TextLine('L%d' % i, 60, 140 + i * 20, 210, 152 + i * 20, 10))
    lines.append(layout.TextLine('centred title', 200, 100, 360, 112, 10))

    order = [line.text for line in layout.reading_order(
        layout.Page(1, 595, 842, tuple(lines)), rtl=True)]
    assert order[0] == 'centred title'
    assert order.index('R0') < order.index('L0')
    assert order.index('R5') < order.index('L0')
