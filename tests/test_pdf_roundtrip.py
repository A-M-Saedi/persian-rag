
import pytest

fitz = pytest.importorskip('fitz')

from news_qa.sources import load_file


BODY = [
    'The distribution network in the northern provinces was expanded',
    'during the past year and the number of subscribers reached a',
    'record level. Growth was concentrated in the residential sector',
    'and the smaller cities accounted for the largest share of it.',
]


@pytest.fixture(scope='module')
def report(tmp_path_factory):
    doc = fitz.open()
    for number in (1, 2, 3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((180, 40), 'Annual Report of the Utility', fontsize=9)
        page.insert_text((285, 810), 'Page %d' % number, fontsize=9)
        for index, text in enumerate(BODY):
            page.insert_text((60, 140 + index * 18), text, fontsize=10)
        page.insert_text((60, 300), 'A long line of ordinary body text that reaches the margin.', fontsize=10)
        page.insert_text((60, 318), 'The distri-', fontsize=10)
        page.insert_text((60, 336), 'bution grid was audited by the regional office.', fontsize=10)
        if number == 1:
            page.insert_text((60, 700), 'This paragraph does not finish on this page and', fontsize=10)
    doc[1].insert_text((60, 120), 'continues right here on the next one.', fontsize=10)
    doc.set_metadata({'title': 'Annual Utility Report'})
    path = tmp_path_factory.mktemp('pdf') / 'report.pdf'
    doc.save(str(path))
    doc.close()
    return path


def test_the_title_comes_from_the_metadata(report):
    assert load_file(report).title == 'Annual Utility Report'


def test_the_source_id_is_a_file_uri(report):
    assert load_file(report).source_id.startswith('file://')


def test_no_page_markers_reach_the_index(report):
    content = load_file(report).content
    assert 'Page 1' not in content
    assert '--- Page' not in content


def test_the_running_header_and_footer_are_removed(report):
    content = load_file(report).content
    assert 'Annual Report of the Utility' not in content
    assert 'Page 2' not in content


def test_a_word_broken_across_two_lines_is_rejoined(report):
    content = load_file(report).content
    assert 'distribution grid was audited' in content
    assert 'distri-' not in content


def test_a_paragraph_split_across_a_page_break_is_stitched(report):
    content = load_file(report).content
    assert 'does not finish on this page and continues right here' in content


def test_an_encrypted_pdf_fails_as_a_source_error(tmp_path):
    from news_qa.sources import SourceError
    doc = fitz.open()
    doc.new_page().insert_text((60, 100), 'secret')
    path = tmp_path / 'enc.pdf'
    doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256,
             owner_pw='o', user_pw='u')
    doc.close()
    with pytest.raises(SourceError):
        load_file(path)


def test_a_mostly_scanned_pdf_reports_instead_of_raising(tmp_path):
    doc = fitz.open()
    for _ in range(4):
        doc.new_page(width=595, height=842)
    doc.new_page(width=595, height=842).insert_text(
        (60, 100), 'Only this page carries any selectable text at all.', fontsize=11)
    path = tmp_path / 'scan.pdf'
    doc.save(str(path))
    doc.close()

    document = load_file(path)
    assert 'selectable text' in document.content
    assert any('OCR' in note for note in document.notes)
