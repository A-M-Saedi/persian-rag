from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TextLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float = 0.0
    bold: bool = False

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass(frozen=True)
class Page:
    number: int
    width: float
    height: float
    lines: Tuple[TextLine, ...] = ()


BODY = 'body'
HEADING = 'heading'
LIST = 'list'
TABLE = 'table'


@dataclass(frozen=True)
class Paragraph:
    text: str
    kind: str = BODY


_INVISIBLE = {
    ' ': ' ',
    '­': '',
    '​': '',
    '‎': '',
    '‏': '',
    '‪': '',
    '‫': '',
    '‬': '',
    '‭': '',
    '‮': '',
    '﻿': '',
}

_INVISIBLE_TABLE = {ord(k): v for k, v in _INVISIBLE.items()}


_ARABIC_LETTER = r'\u0620-\u064A\u0671-\u06D3'
_LTR_RUN = r'0-9A-Za-z\u0660-\u0669\u06F0-\u06F9'
_GLUED = (
    re.compile(r'([%s])(?=[%s])' % (_ARABIC_LETTER, _LTR_RUN)),
    re.compile(r'(?<=[%s])(?=[%s])' % (_LTR_RUN, _ARABIC_LETTER)),
)


def split_bidi_runs(text: str) -> str:
    for pattern in _GLUED:
        text = pattern.sub(lambda m: (m.group(0) + ' '), text)
    return text


def clean_unicode(text: str) -> str:
    text = text.translate(_INVISIBLE_TABLE)
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = split_bidi_runs(text)
    return text.strip()


_RTL_RANGES = (
    (0x0590, 0x05FF),
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB1D, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def _is_rtl_char(ch: str) -> bool:
    code = ord(ch)
    return any(low <= code <= high for low, high in _RTL_RANGES)


def rtl_ratio(text: str) -> float:
    rtl = sum(1 for ch in text if _is_rtl_char(ch))
    ltr = sum(1 for ch in text if ch.isalpha() and not _is_rtl_char(ch))
    total = rtl + ltr
    return rtl / total if total else 0.0


def is_rtl(text: str) -> bool:
    return rtl_ratio(text) > 0.5


def join_spans(spans: Sequence[Tuple[str, float, float]], rtl: bool) -> str:
    cleaned = [
        (clean_unicode(text), x0, x1)
        for text, x0, x1 in spans
        if clean_unicode(text)
    ]
    if not cleaned:
        return ''

    cleaned.sort(key=lambda span: span[1], reverse=rtl)

    out = cleaned[0][0]
    previous = cleaned[0]
    for text, x0, x1 in cleaned[1:]:
        scale = (previous[2] - previous[1]) / max(1, len(previous[0]))
        gap = (previous[1] - x1) if rtl else (x0 - previous[2])
        if out.endswith(' ') or text.startswith(' '):
            out = f'{out}{text}'
        elif gap > 0.25 * scale:
            out = f'{out} {text}'
        else:
            out = f'{out}{text}'
        previous = (text, x0, x1)

    return re.sub(r'[^\S\n]+', ' ', out).strip()


FURNITURE_ZONE = 0.12

FURNITURE_PAGE_RATIO = 0.5

FURNITURE_MIN_PAGES = 3

_DIGITS = re.compile(r'[0-9٠-٩۰-۹]+')


def furniture_key(text: str) -> str:
    return _DIGITS.sub('#', clean_unicode(text).lower()).strip()


def find_furniture(pages: Sequence[Page]) -> frozenset:
    if len(pages) < FURNITURE_MIN_PAGES:
        return frozenset()

    seen_on = {}
    for page in pages:
        if page.height <= 0:
            continue
        top = page.height * FURNITURE_ZONE
        bottom = page.height * (1 - FURNITURE_ZONE)
        keys = {
            furniture_key(line.text)
            for line in page.lines
            if line.y1 <= top or line.y0 >= bottom
        }
        for key in keys:
            if key:
                seen_on.setdefault(key, set()).add(page.number)

    needed = max(2, round(len(pages) * FURNITURE_PAGE_RATIO))
    return frozenset(key for key, pages_seen in seen_on.items() if len(pages_seen) >= needed)


def drop_furniture(page: Page, furniture: frozenset) -> Page:
    if not furniture or page.height <= 0:
        return page

    top = page.height * FURNITURE_ZONE
    bottom = page.height * (1 - FURNITURE_ZONE)
    kept = tuple(
        line for line in page.lines
        if not (
            (line.y1 <= top or line.y0 >= bottom)
            and furniture_key(line.text) in furniture
        )
    )
    return Page(page.number, page.width, page.height, kept)


def drop_repeated_lines(lines: Sequence[TextLine]) -> List[TextLine]:
    out: List[TextLine] = []
    seen = set()
    for line in lines:
        key = (furniture_key(line.text), round(line.x0, 0), round(line.y0, 0))
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


WIDE_LINE_RATIO = 0.6

MIN_GUTTER_RATIO = 0.05


GUTTER_CROSSING_RATIO = 0.25
GUTTER_MARGIN = 0.2

MIN_COLUMN_LINES = 3

MIN_COLUMN_HEIGHT_RATIO = 0.4

_MAX_COLUMN_DEPTH = 3


def _sorted_by_position(lines: Iterable[TextLine], rtl: bool) -> List[TextLine]:
    return sorted(
        lines,
        key=lambda line: (round(line.y0, 1), -line.x1 if rtl else line.x0),
    )


def find_gutter(lines: Sequence[TextLine]) -> Optional[float]:
    if len(lines) < MIN_COLUMN_LINES * 2:
        return None

    left = min(line.x0 for line in lines)
    right = max(line.x1 for line in lines)
    span = right - left
    if span <= 0:
        return None

    min_width = max(span * MIN_GUTTER_RATIO, 8.0)
    max_crossing = int(len(lines) * GUTTER_CROSSING_RATIO)

    cells = 240
    step = span / cells
    profile = [0] * cells
    for line in lines:
        first = max(0, int((line.x0 - left) / step))
        last = min(cells - 1, int((line.x1 - left) / step))
        for index in range(first, last + 1):
            profile[index] += 1

    best_width, best_center = 0.0, None
    run_start = None
    for index in range(cells + 1):
        quiet = index < cells and profile[index] <= max_crossing
        if quiet and run_start is None:
            run_start = index
        elif not quiet and run_start is not None:
            start_x = left + run_start * step
            end_x = left + index * step
            width = end_x - start_x
            center = (start_x + end_x) / 2.0
            inside = (
                left + span * GUTTER_MARGIN <= center <= right - span * GUTTER_MARGIN
            )
            if inside and width > best_width:
                best_width, best_center = width, center
            run_start = None

    if best_center is None or best_width < min_width:
        return None
    return best_center


def _split_at_gutter(
    lines: Sequence[TextLine], gutter: float
) -> Optional[Tuple[List[TextLine], List[TextLine]]]:
    left = [line for line in lines if line.x1 <= gutter]
    right = [line for line in lines if line.x0 >= gutter]
    if len(left) + len(right) != len(lines):
        return None
    if len(left) < MIN_COLUMN_LINES or len(right) < MIN_COLUMN_LINES:
        return None

    total = max(line.y1 for line in lines) - min(line.y0 for line in lines)
    if total <= 0:
        return None
    for side in (left, right):
        extent = max(line.y1 for line in side) - min(line.y0 for line in side)
        if extent < total * MIN_COLUMN_HEIGHT_RATIO:
            return None
    return left, right


def _split_at_crossers(
    lines: Sequence[TextLine], crossers: set, rtl: bool, depth: int
) -> List[TextLine]:
    ordered: List[TextLine] = []
    band: List[TextLine] = []
    for line in _sorted_by_position(lines, rtl):
        if id(line) in crossers:
            ordered.extend(_order_group(band, rtl, depth + 1))
            band = []
            ordered.append(line)
        else:
            band.append(line)
    ordered.extend(_order_group(band, rtl, depth + 1))
    return ordered


def _order_group(lines: Sequence[TextLine], rtl: bool, depth: int = 0) -> List[TextLine]:
    if depth >= _MAX_COLUMN_DEPTH or len(lines) < MIN_COLUMN_LINES * 2:
        return _sorted_by_position(lines, rtl)

    gutter = find_gutter(lines)
    if gutter is None:
        return _sorted_by_position(lines, rtl)

    crossing = [line for line in lines if line.x0 < gutter < line.x1]
    if crossing:
        return _split_at_crossers(lines, set(map(id, crossing)), rtl, depth)

    split = _split_at_gutter(lines, gutter)
    if split is None:
        return _sorted_by_position(lines, rtl)

    left, right = split
    first, second = (right, left) if rtl else (left, right)
    return (
        _order_group(first, rtl, depth + 1)
        + _order_group(second, rtl, depth + 1)
    )


def reading_order(page: Page, rtl: bool) -> List[TextLine]:
    lines = drop_repeated_lines(_sorted_by_position(page.lines, rtl))
    if not lines:
        return []

    content_width = max(
        page.width,
        max(line.x1 for line in lines) - min(line.x0 for line in lines),
    )
    wide = content_width * WIDE_LINE_RATIO

    ordered: List[TextLine] = []
    band: List[TextLine] = []
    for line in lines:
        if line.width >= wide:
            ordered.extend(_order_group(band, rtl))
            band = []
            ordered.append(line)
        else:
            band.append(line)
    ordered.extend(_order_group(band, rtl))
    return ordered


_LIST_MARKER = re.compile(
    r'^\s*(?:'
    r'[-*•▪◦‣–—]\s+'
    r'|\(?[0-9٠-٩۰-۹]{1,3}[.)\-]\s+'
    r'|[A-Za-z][.)]\s+'
    r'|[ا-ی][.)]\s+'
    r')'
)

MAX_HEADING_WORDS = 16

HEADING_SIZE_RATIO = 1.15

BOLD_IS_MEANINGFUL_BELOW = 0.3

_SENTENCE_END = ('.', '؟', '!', '?', ':', '؛', ';', '…')

_WORD_CONTINUES = re.compile(r'([A-Za-z][-‐]|\u200c)$')


def word_continues(text: str) -> bool:
    return bool(_WORD_CONTINUES.search(text.rstrip()))


def is_list_item(text: str) -> bool:
    return bool(_LIST_MARKER.match(text))


def is_heading(line: TextLine, body_size: float, bold_matters: bool = True) -> bool:
    text = line.text.strip()
    if not text or len(text.split()) > MAX_HEADING_WORDS:
        return False
    if text.endswith(('.', '،', '؛', ',')):
        return False
    if body_size > 0 and line.size >= body_size * HEADING_SIZE_RATIO:
        return True
    if bold_matters and line.bold and body_size > 0 and line.size >= body_size:
        return True

    if rtl_ratio(text) < 0.5:
        letters = re.sub(r'[^A-Za-z]', '', text)
        return bool(letters and len(letters) >= 4 and letters.isupper() and len(text) <= 80)
    return False


SHORT_LINE_RATIO = 0.72

VERY_SHORT_LINE_RATIO = 0.5

INDENT_RATIO = 0.02

SIZE_CHANGE_RATIO = 0.15


def median(values: Iterable[float], default: float = 0.0) -> float:
    values = [v for v in values if v > 0]
    return statistics.median(values) if values else default


def join_lines(texts: Sequence[str]) -> str:
    if not texts:
        return ''

    out = texts[0].strip()
    for raw in texts[1:]:
        text = raw.strip()
        if not text:
            continue
        if not out:
            out = text
            continue
        if re.search(r'[A-Za-z][-‐]$', out) and re.match(r'^[a-z]', text):
            out = out[:-1] + text
        elif out.endswith('‌'):
            out = out + text
        else:
            out = f'{out} {text}'
    return out


def lines_to_paragraphs(lines: Sequence[TextLine], rtl: bool) -> List[Paragraph]:
    if not lines:
        return []

    body_size = median((line.size for line in lines), default=0.0)
    line_height = median((line.height for line in lines), default=10.0)
    content_width = max(line.width for line in lines) or 1.0
    left_margin = min(line.x0 for line in lines)
    right_margin = max(line.x1 for line in lines)
    bold_matters = (
        sum(1 for line in lines if line.bold) / len(lines) < BOLD_IS_MEANINGFUL_BELOW
    )

    gaps = [
        max(0.0, current.y0 - previous.y1)
        for previous, current in zip(lines, lines[1:])
    ]
    leading = statistics.median(gaps) if gaps else 0.0
    gap_limit = leading + max(line_height * 0.5, 1.5)
    indent = max(content_width * INDENT_RATIO, 6.0)

    paragraphs: List[Paragraph] = []
    buffer: List[str] = []
    buffer_kind = BODY

    def flush():
        if buffer:
            text = join_lines(buffer)
            if text:
                paragraphs.append(Paragraph(text, buffer_kind))
        buffer.clear()

    previous: Optional[TextLine] = None
    for line in lines:
        heading = is_heading(line, body_size, bold_matters)
        listed = is_list_item(line.text)
        kind = HEADING if heading else LIST if listed else BODY

        if previous is None:
            starts_paragraph = True
        else:
            previous_heading = is_heading(previous, body_size, bold_matters)
            previous_width = previous.width
            ends_sentence = previous.text.rstrip().endswith(_SENTENCE_END)
            unfinished = word_continues(previous.text)
            if rtl:
                indented = (
                    right_margin - line.x1 > indent
                    and line.x0 <= left_margin + content_width * 0.1
                )
            else:
                indented = (
                    line.x0 - left_margin > indent
                    and line.x1 >= right_margin - content_width * 0.1
                )
            size_jump = (
                body_size > 0
                and previous.size > 0
                and line.size > 0
                and abs(line.size - previous.size) / body_size > SIZE_CHANGE_RATIO
            )
            starts_paragraph = (
                heading
                or listed
                or previous_heading
                or size_jump
                or (not unfinished and indented)
                or (line.y0 - previous.y1) > gap_limit
                or (not unfinished and previous_width < content_width * VERY_SHORT_LINE_RATIO)
                or (
                    not unfinished
                    and previous_width < content_width * SHORT_LINE_RATIO
                    and ends_sentence
                )
            )

        if starts_paragraph:
            flush()
            buffer_kind = kind
        buffer.append(line.text)
        previous = line

    flush()
    return paragraphs


def _continues(previous: Paragraph, following: Paragraph) -> bool:
    if previous.kind != BODY or following.kind != BODY:
        return False
    if not previous.text or not following.text:
        return False
    if previous.text.rstrip().endswith(_SENTENCE_END + ('»', '"', ')')):
        return False
    return not following.text[:1].isupper()


def merge_pages(per_page: Sequence[Sequence[Paragraph]]) -> List[Paragraph]:
    merged: List[Paragraph] = []
    for page_paragraphs in per_page:
        for index, paragraph in enumerate(page_paragraphs):
            if index == 0 and merged and _continues(merged[-1], paragraph):
                joined = join_lines([merged[-1].text, paragraph.text])
                merged[-1] = Paragraph(joined, merged[-1].kind)
                continue
            merged.append(paragraph)
    return merged


def document_paragraphs(pages: Sequence[Page], rtl: Optional[bool] = None) -> List[Paragraph]:
    if rtl is None:
        rtl = is_rtl(' '.join(line.text for page in pages for line in page.lines))

    furniture = find_furniture(pages)
    return merge_pages([
        lines_to_paragraphs(reading_order(drop_furniture(page, furniture), rtl), rtl)
        for page in pages
    ])


def render(paragraphs: Sequence[Paragraph]) -> str:
    return '\n\n'.join(p.text for p in paragraphs if p.text.strip())


def render_table(rows: Sequence[Sequence[Optional[str]]], rtl: bool) -> List[str]:
    cleaned = [
        [clean_unicode(str(cell)) if cell is not None else '' for cell in row]
        for row in rows
    ]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return []

    if rtl:
        cleaned = [list(reversed(row)) for row in cleaned]

    header = cleaned[0]
    has_header = len(cleaned) > 1 and all(cell.strip() for cell in header)

    out: List[str] = []
    if not has_header:
        return [' | '.join(cell for cell in row if cell) for row in cleaned]

    out.append(' | '.join(header))
    for row in cleaned[1:]:
        pairs = [
            f'{name}: {value}'
            for name, value in zip(header, row)
            if value.strip()
        ]
        if pairs:
            out.append(' | '.join(pairs))
    return out
