from news_qa.cli import (
    DEBUG_WORDS,
    EXIT_WORDS,
    HELP_WORDS,
    MODE_WORDS,
    OPEN_WORDS,
    EXTRACT_PREVIEW_CHARS,
    PREVIEW_CHARS,
    STRICT_WORDS,
    format_extraction,
    resolve_source_choice,
    format_retrieval,
    is_affirmative,
    resolve_policy_choice,
)
from news_qa.policy import OPEN, STRICT
from news_qa.rag import Retrieved
from news_qa.document import KIND_PDF, KIND_WEB, Document
from news_qa.scraper import THIN_ARTICLE_WORDS

KEPT = Retrieved('جواد نکونام سرمربی تراکتور شد.', 0.378, True)
BLOCKED = Retrieved('پادکست ورزش سه برگزار شد.', 0.632, False)
LEXICAL = Retrieved('تراکتور در تبریز بازی می‌کند.', 0.667, True, 'lexical', 2.044)
TRIMMED = Retrieved('یک نامزدِ قبول‌شدهٔ دیگر.', 0.44, True, 'dense', 0.0, False)
UNTRACED = Retrieved('زبان مردم این منطقه فارسی است.', 0.550, False)


def test_debug_view_shows_the_distance_of_every_candidate():
    view = format_retrieval([KEPT, BLOCKED], 0.60)
    assert '0.378' in view
    assert '0.632' in view


def test_debug_view_marks_kept_and_blocked_differently():
    kept_line, blocked_line = format_retrieval([KEPT, BLOCKED], 0.60).splitlines()[1:3]
    assert '✅' in kept_line and '⛔️' not in kept_line
    assert '⛔️' in blocked_line and '✅' not in blocked_line


def test_debug_view_names_the_channel_that_found_each_chunk():
    view = format_retrieval([KEPT, LEXICAL], 0.60)
    kept_line, lexical_line = view.splitlines()[1:3]

    assert 'برداری' in kept_line
    assert 'واژگانی' in lexical_line
    assert '2.04' in lexical_line, 'امتیاز واژگانی باید دیده شود، نه فقط نامِ کانال'


def test_a_lexical_candidate_can_be_kept_past_the_threshold():
    view = format_retrieval([LEXICAL], 0.60)
    assert '✅' in view
    assert 'دورتر از آستانه' not in view


def test_debug_view_explains_how_far_past_the_threshold_a_total_miss_was():
    view = format_retrieval([BLOCKED], 0.60)
    assert '0.032' in view, 'فاصله تا آستانه باید گزارش شود، نه فقط خودِ فاصله'


def test_no_distance_gap_is_reported_when_something_did_pass():
    assert 'دورتر از آستانه' not in format_retrieval([KEPT, BLOCKED], 0.60)


def test_debug_view_survives_an_empty_candidate_list():
    assert format_retrieval([], 0.60)


def test_long_chunks_are_truncated_and_short_ones_are_not():
    long_doc = Retrieved('الف' * 200, 0.4, True)
    short_doc = Retrieved('الف', 0.4, True)
    assert '…' in format_retrieval([long_doc], 0.60)
    assert '…' not in format_retrieval([short_doc], 0.60)


def test_newlines_in_a_chunk_do_not_break_the_line_layout():
    multiline = Retrieved('خط اول\nخط دوم\n\nپاراگراف بعدی', 0.4, True)
    assert len(format_retrieval([multiline], 0.60).splitlines()) == 2


def test_preview_length_is_respected():
    doc = Retrieved('ژ' * (PREVIEW_CHARS * 2), 0.4, True)
    line = format_retrieval([doc], 0.60).splitlines()[1]
    assert line.count('ژ') == PREVIEW_CHARS


def test_command_words_do_not_collide_with_each_other():
    groups = [
        set(EXIT_WORDS), set(DEBUG_WORDS), set(HELP_WORDS), set(MODE_WORDS),
    ]
    for i, first in enumerate(groups):
        for second in groups[i + 1:]:
            assert not first & second


def test_command_words_are_lowercase():
    for word in EXIT_WORDS + DEBUG_WORDS + HELP_WORDS + MODE_WORDS:
        assert word == word.lower()


# --------------------------------------------------------- انتخاب سیاست پاسخ


def test_persian_digits_select_the_same_mode_as_latin_digits():
    assert resolve_policy_choice('۱', OPEN) is STRICT
    assert resolve_policy_choice('۲', STRICT) is OPEN
    assert resolve_policy_choice('۲', STRICT) is resolve_policy_choice('2', STRICT)


def test_empty_input_keeps_the_current_mode():
    assert resolve_policy_choice('', OPEN) is OPEN
    assert resolve_policy_choice('   ', STRICT) is STRICT


def test_an_unrecognized_answer_never_silently_selects_a_mode():
    for junk in ['چرت', '3', 'y', 'strictt', '۳']:
        assert resolve_policy_choice(junk, STRICT) is None, junk
        assert resolve_policy_choice(junk, OPEN) is None, junk


def test_both_spellings_of_the_strict_word_are_accepted():
    assert resolve_policy_choice('سخت‌گیرانه', OPEN) is STRICT
    assert resolve_policy_choice('سختگیرانه', OPEN) is STRICT


def test_mode_words_do_not_collide_with_the_policy_words():
    assert not set(MODE_WORDS) & (set(STRICT_WORDS) | set(OPEN_WORDS))
    assert not set(STRICT_WORDS) & set(OPEN_WORDS)


def test_only_an_explicit_yes_confirms_an_ungrounded_answer():
    assert is_affirmative('بله')
    assert is_affirmative('y')
    assert not is_affirmative('')
    assert not is_affirmative('خیر')
    assert not is_affirmative('nope')


def test_the_debug_view_says_when_an_answer_had_no_article_text():
    assert 'بدون هیچ متنی' in format_retrieval([BLOCKED], 0.60, OPEN)
    assert 'بدون هیچ متنی' not in format_retrieval([BLOCKED], 0.60, STRICT)
    assert 'بدون هیچ متنی' not in format_retrieval([BLOCKED], 0.60)


def test_the_no_article_text_note_only_appears_when_nothing_passed():
    assert 'بدون هیچ متنی' not in format_retrieval([KEPT, BLOCKED], 0.60, OPEN)


def test_debug_view_distinguishes_a_budget_cut_from_a_gate_rejection():
    view = format_retrieval([KEPT, TRIMMED, BLOCKED], 0.60, None, 1)
    kept_line, trimmed_line, blocked_line = view.splitlines()[1:4]

    assert '✅' in kept_line
    assert '✅' not in trimmed_line and '⛔️' not in trimmed_line
    assert '⛔️' in blocked_line
    assert 'بودجه' in view and '(1)' in view


def test_the_budget_line_stays_quiet_when_nothing_was_trimmed():
    assert 'بودجهٔ پرامپت' not in format_retrieval([KEPT, BLOCKED], 0.60, None, 6)


def test_the_debug_view_separates_a_coverage_rejection_from_a_distance_rejection():
    view = format_retrieval([UNTRACED, BLOCKED], threshold=0.63)

    assert "⛔️ پوشش" in view
    assert "⛔️ رد" in view
    assert "هیچ واژهٔ محتوایی" in view


def test_the_nearest_candidate_line_never_reports_a_negative_distance():
    view = format_retrieval([UNTRACED], threshold=0.63)

    assert "دورتر از آستانه" not in view
    assert "-0." not in view


def test_every_retrieval_marker_has_the_same_width():
    rows = format_retrieval(
        [KEPT, BLOCKED, TRIMMED, UNTRACED], threshold=0.63, budget=2
    ).splitlines()[1:5]

    positions = {row.replace('\ufe0f', '').index('فاصله') for row in rows}
    assert len(positions) == 1, f"ستونِ فاصله هم‌تراز نیست: {positions}"


THIN = Document('تماس با ما', 'نشانی: تهران - خیابان میرزای شیرازی - پلاک ۲۵',
                'https://example.com/a', KIND_WEB)
FULL = Document('یک بمب خبری دیگر در تراکتور!', 'به گزارش ورزش سه ' * 60,
                'https://example.com/b', KIND_WEB)


def test_the_extraction_view_shows_the_text_that_was_actually_stored():
    view = format_extraction(THIN)

    assert 'میرزای شیرازی' in view, "متنِ استخراج‌شده در نما نیست"


def test_a_thin_extraction_is_flagged_and_a_full_article_is_not():
    view_thin, view_full = format_extraction(THIN), format_extraction(FULL)

    assert '⚠️' in view_thin
    assert '⚠️' not in view_full


def test_the_thin_warning_never_claims_the_text_is_too_short_to_answer():
    view = format_extraction(THIN)

    assert 'استخراج' in view
    for claim in ('پاسخ', 'سوال', 'کوتاه است'):
        assert claim not in view.split('⚠️')[1], f"هشدار دربارهٔ «{claim}» ادعا می‌کند"


def test_the_preview_is_truncated_and_never_breaks_across_lines():
    article = Document('ت', 'الف\n\nب\n' + 'ژ' * (EXTRACT_PREVIEW_CHARS * 2),
                       'https://example.com/c', KIND_WEB)

    preview = [ln for ln in format_extraction(article).splitlines() if ln.startswith('📝')]

    assert len(preview) == 1
    assert preview[0].endswith('…')
    assert len(preview[0].split(' ', 1)[1]) == EXTRACT_PREVIEW_CHARS + 1


def test_the_thin_threshold_separates_every_recorded_fixture(baseline):
    thin = {
        name: row['content_words'] < THIN_ARTICLE_WORDS
        for name, row in baseline.items()
    }

    assert thin == {
        'hamshahri_1063549': True,
        'khabaronline_1940521': True,
        'varzesh3_2410970': False,
        'zoomit_465572': False,
        'yjc_9118143': False,
        'mehr_6914285': False,
        'wikipedia_persian_gulf': False,
    }
    assert baseline['khabaronline_1940521']['content_words'] > \
        baseline['hamshahri_1063549']['content_words'], \
        "اگر این وارونه شود، استدلالِ «طول جدا نمی‌کند» دیگر پشتوانه ندارد"


PDF_DOC = Document('گزارش سالانه', 'واژه ' * 300, 'file:///tmp/گزارش.pdf', KIND_PDF,
                   notes=('۲ جدول شناسایی و بازنویسی شد.',))


def test_the_source_choice_understands_both_scripts_and_both_languages():
    for answer in ('1', '۱', 'file', 'فایل'):
        assert resolve_source_choice(answer) == 'file', answer
    for answer in ('2', '۲', 'url', 'آدرس'):
        assert resolve_source_choice(answer) == 'url', answer


def test_an_empty_source_choice_is_not_silently_defaulted():
    assert resolve_source_choice('') is None
    assert resolve_source_choice('شاید') is None


def test_the_extraction_notes_are_shown_after_the_warning_not_instead_of_it():
    view = format_extraction(PDF_DOC)

    assert '⚠️' not in view, 'سندِ ۳۰۰ واژه‌ای نازک نیست'
    assert 'جدول' in view


def test_a_file_warning_never_mentions_scraping_a_page():
    thin_file = Document('سند', 'سه واژه دارد', 'file:///tmp/x.pdf', KIND_PDF)

    warning = format_extraction(thin_file).split('⚠️')[1]

    assert 'صفحه' not in warning
    assert 'فایل' in warning


def test_the_thin_warning_stays_silent_about_answer_quality_for_files_too():
    warning = format_extraction(
        Document('سند', 'سه واژه دارد', 'file:///tmp/x.pdf', KIND_PDF)
    ).split('⚠️')[1]

    for claim in ('پاسخ', 'سوال', 'کوتاه است'):
        assert claim not in warning
