from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..document import KIND_PDF, Document, SourceError
from . import layout
from .layout import Page, Paragraph, TextLine, TABLE

_BOLD_FLAG = 1 << 4

MIN_CHARS_PER_PAGE = 40

SCANNED_PAGE_RATIO = 0.5

LARGE_DOCUMENT_PAGES = 200

_JUNK_TITLE_HINTS = ('untitled', 'microsoft word', 'document1', '.doc', '.pdf', '.tex')

_CEREMONIAL_TITLES = (
    'بسمه تعالی', 'باسمه تعالی', 'بسم الله', 'به نام خدا', 'بنام خدا',
    'هو الحق', 'هوالحق',
)


def _is_junk_title(text: str) -> bool:
    stripped = text.strip().strip('"«»\'').strip()
    lowered = stripped.lower()
    if any(hint in lowered for hint in _JUNK_TITLE_HINTS):
        return True
    compact = ' '.join(stripped.replace('\u200c', ' ').split())
    return any(compact.startswith(head) for head in _CEREMONIAL_TITLES)


def _require_fitz():
    try:
        import fitz
    except ImportError as exc:
        raise SourceError(
            'برای خواندن PDF کتابخانهٔ PyMuPDF لازم است. نصب: '
            'python -m pip install -r requirements-pdf.txt'
        ) from exc
    return fitz


def _span_is_bold(span: dict) -> bool:
    if int(span.get('flags', 0)) & _BOLD_FLAG:
        return True
    font = str(span.get('font', '')).lower()
    return 'bold' in font or 'black' in font


def page_lines(raw: dict, rtl: bool) -> List[TextLine]:
    lines: List[TextLine] = []
    for block in raw.get('blocks', ()):
        if block.get('type') != 0:
            continue
        for raw_line in block.get('lines', ()):
            spans = [
                (str(span.get('text', '')), float(span['bbox'][0]), float(span['bbox'][2]))
                for span in raw_line.get('spans', ())
                if span.get('bbox')
            ]
            text = layout.join_spans(spans, rtl)
            if not text:
                continue

            bbox = raw_line.get('bbox') or block.get('bbox')
            if not bbox:
                continue
            x0, y0, x1, y1 = (float(value) for value in bbox)

            sizes = [
                float(span.get('size') or 0.0)
                for span in raw_line.get('spans', ())
                if float(span.get('size') or 0.0) > 0
            ]
            bold_spans = [span for span in raw_line.get('spans', ()) if _span_is_bold(span)]
            all_spans = list(raw_line.get('spans', ()))

            lines.append(TextLine(
                text=text,
                x0=x0, y0=y0, x1=x1, y1=y1,
                size=max(sizes) if sizes else 0.0,
                bold=bool(all_spans) and len(bold_spans) == len(all_spans),
            ))
    return lines


def _table_bands(page, fitz) -> List[Tuple[float, float, List[List[Optional[str]]]]]:
    finder = getattr(page, 'find_tables', None)
    if finder is None:
        return []
    try:
        found = finder()
        tables = list(getattr(found, 'tables', found) or ())
    except Exception:
        return []

    bands = []
    for table in tables:
        try:
            rows = table.extract()
            bbox = table.bbox
        except Exception:
            continue
        if not rows:
            continue
        bands.append((float(bbox[1]), float(bbox[3]), rows))
    bands.sort(key=lambda band: band[0])
    return bands


def _split_by_bands(
    lines: Sequence[TextLine], bands: Sequence[Tuple[float, float, list]]
) -> List[Tuple[Optional[float], List[TextLine]]]:
    if not bands:
        return [(None, list(lines))]

    groups: List[Tuple[Optional[float], List[TextLine]]] = []
    remaining = list(lines)
    for y0, y1, _ in bands:
        before = [line for line in remaining if line.y1 <= y0]
        groups.append((y0, before))
        remaining = [line for line in remaining if line.y0 >= y1]
    groups.append((None, remaining))
    return groups


def _title_from_metadata(metadata: dict) -> str:
    title = layout.clean_unicode(str((metadata or {}).get('title') or ''))
    if not title or len(title) < 3:
        return ''
    return '' if _is_junk_title(title) else title


def title_from_first_page(lines: Sequence[TextLine]) -> str:
    if not lines:
        return ''
    top = sorted(lines, key=lambda line: line.y0)[: max(12, len(lines) // 4)]
    body = layout.median((line.size for line in lines), default=0.0)
    candidates = [
        line for line in top
        if line.size > body * layout.HEADING_SIZE_RATIO
        and 0 < len(line.text.split()) <= layout.MAX_HEADING_WORDS
    ]
    candidates = [line for line in candidates if not _is_junk_title(line.text)]
    if not candidates:
        return ''
    best = max(candidates, key=lambda line: (line.size, -line.y0))
    return best.text.strip().strip('"«»\u201c\u201d').strip()


def load(path: Path) -> Document:
    fitz = _require_fitz()

    try:
        document = fitz.open(path)
    except Exception as exc:
        raise SourceError(f'فایل PDF باز نشد: {exc}') from exc

    try:
        if document.is_encrypted and not document.authenticate(''):
            raise SourceError(f'فایل PDF رمزدار است: {path.name}')
        if document.page_count == 0:
            raise SourceError(f'فایل PDF هیچ صفحه‌ای ندارد: {path.name}')

        raw_pages = []
        for index, page in enumerate(document, start=1):
            raw_pages.append((
                index,
                page.rect,
                page.get_text('dict', flags=fitz.TEXTFLAGS_TEXT),
                _table_bands(page, fitz),
            ))
        metadata = dict(document.metadata or {})
    finally:
        document.close()

    sample = ' '.join(
        str(span.get('text', ''))
        for _, _, raw, _ in raw_pages
        for block in raw.get('blocks', ())
        if block.get('type') == 0
        for line in block.get('lines', ())
        for span in line.get('spans', ())
    )
    rtl = layout.is_rtl(layout.clean_unicode(sample))

    pages, tables_by_page, empty_pages = [], {}, []
    for number, rect, raw, bands in raw_pages:
        lines = page_lines(raw, rtl)
        if sum(len(line.text) for line in lines) < MIN_CHARS_PER_PAGE:
            empty_pages.append(number)
        pages.append(Page(number, float(rect.width), float(rect.height), tuple(lines)))
        tables_by_page[number] = bands

    furniture = layout.find_furniture(pages)

    per_page, table_count = [], 0
    for page in pages:
        ordered = layout.reading_order(layout.drop_furniture(page, furniture), rtl)
        bands = tables_by_page[page.number]

        page_paragraphs: List[Paragraph] = []
        for anchor, group in _split_by_bands(ordered, bands):
            page_paragraphs.extend(layout.lines_to_paragraphs(group, rtl))
            if anchor is None:
                continue
            rows = next(rows for y0, _, rows in bands if y0 == anchor)
            rendered = layout.render_table(rows, rtl)
            if rendered:
                table_count += 1
                page_paragraphs.extend(Paragraph(row, TABLE) for row in rendered)
        per_page.append(page_paragraphs)

    content = layout.render(layout.merge_pages(per_page))
    notes = _notes(len(pages), empty_pages, table_count, len(content.split()))

    title = (
        _title_from_metadata(metadata)
        or title_from_first_page(pages[0].lines if pages else ())
        or _title_from_filename(path)
    )
    return Document(
        title=title,
        content=content,
        source_id=path.resolve().as_uri(),
        kind=KIND_PDF,
        notes=notes,
    )


def _title_from_filename(path: Path) -> str:
    stem = path.stem.replace('_', ' ').replace('-', ' ')
    return ' '.join(stem.split()) or path.name


def _notes(page_count: int, empty_pages: Sequence[int], tables: int, words: int) -> Tuple[str, ...]:
    notes: List[str] = []

    if empty_pages and len(empty_pages) >= max(1, page_count * SCANNED_PAGE_RATIO):
        notes.append(
            f'از {page_count} صفحه، {len(empty_pages)} صفحه متنِ قابلِ انتخاب نداشت. '
            'این فایل احتمالاً اسکن است و برای استخراج به OCR نیاز دارد.'
        )
    elif empty_pages:
        preview = '، '.join(str(number) for number in empty_pages[:8])
        more = ' و چند صفحهٔ دیگر' if len(empty_pages) > 8 else ''
        notes.append(f'صفحه‌های بدونِ متنِ قابلِ انتخاب: {preview}{more}.')

    if tables:
        notes.append(
            f'{tables} جدول شناسایی و به‌صورتِ «سرستون: مقدار» بازنویسی شد؛ '
            'آرایشِ اصلیِ جدول حفظ نمی‌شود.'
        )

    if page_count >= LARGE_DOCUMENT_PAGES:
        notes.append(
            f'{page_count} صفحه ({words} کلمه) — نمایه‌سازی ممکن است چند دقیقه طول بکشد.'
        )

    return tuple(notes)
