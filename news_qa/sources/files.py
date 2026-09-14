from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

from ..document import Document, SourceError


def _load_pdf(path: Path) -> Document:
    from . import pdf
    return pdf.load(path)


def _load_text(path: Path) -> Document:
    from . import plaintext
    return plaintext.load(path)


EXTRACTORS: Dict[str, Callable[[Path], Document]] = {
    '.pdf': _load_pdf,
    '.txt': _load_text,
    '.md': _load_text,
    '.markdown': _load_text,
}


def supported_suffixes() -> List[str]:
    return sorted(EXTRACTORS)


def load_file(path) -> Document:
    path = Path(path).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SourceError(f'فایل پیدا نشد: {path}') from exc
    if not resolved.is_file():
        raise SourceError(f'مسیر یک فایل نیست: {resolved}')

    suffix = resolved.suffix.lower()
    extractor = EXTRACTORS.get(suffix)
    if extractor is None:
        known = '، '.join(supported_suffixes())
        raise SourceError(
            f'قالبِ «{suffix or "بدون پسوند"}» پشتیبانی نمی‌شود. قالب‌های موجود: {known}'
        )
    return extractor(resolved)
