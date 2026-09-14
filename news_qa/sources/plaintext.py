from __future__ import annotations

from pathlib import Path

from ..document import KIND_TEXT, Document, SourceError
from .layout import clean_unicode

MAX_BYTES = 32 * 1024 * 1024


def title_from_name(path: Path) -> str:
    stem = path.stem.replace('_', ' ').replace('-', ' ')
    return ' '.join(stem.split()) or path.name


def load(path: Path) -> Document:
    if path.stat().st_size > MAX_BYTES:
        raise SourceError(
            f'فایل بزرگ‌تر از {MAX_BYTES // (1024 * 1024)} مگابایت است: {path.name}'
        )
    try:
        raw = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        raise SourceError(
            f'کدگذاریِ فایل UTF-8 نیست: {path.name}. ابتدا آن را به UTF-8 تبدیل کنید.'
        )

    paragraphs = [clean_unicode(part) for part in raw.split('\n\n')]
    content = '\n\n'.join(part for part in paragraphs if part)

    return Document(
        title=title_from_name(path),
        content=content,
        source_id=path.resolve().as_uri(),
        kind=KIND_TEXT,
    )
