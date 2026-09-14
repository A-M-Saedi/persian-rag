from __future__ import annotations

from ..document import (
    KIND_PDF, KIND_TEXT, KIND_WEB, Document, SourceError,
)
from .files import EXTRACTORS, load_file, supported_suffixes
from .web import load_url

THIN_WORDS = {
    KIND_PDF: 40,
    KIND_TEXT: 40,
}


def thin_threshold(kind: str) -> int:
    from ..scraper import THIN_ARTICLE_WORDS
    return THIN_WORDS.get(kind, THIN_ARTICLE_WORDS)


def is_thin(document: Document) -> bool:
    return document.word_count < thin_threshold(document.kind)


__all__ = [
    'Document', 'SourceError',
    'KIND_WEB', 'KIND_PDF', 'KIND_TEXT',
    'EXTRACTORS', 'load_file', 'load_url', 'supported_suffixes',
    'THIN_WORDS', 'thin_threshold', 'is_thin',
]
