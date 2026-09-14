from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

KIND_WEB = 'web'
KIND_PDF = 'pdf'
KIND_TEXT = 'text'


class SourceError(Exception):
    pass


@dataclass(frozen=True)
class Document:
    title: str
    content: str
    source_id: str
    kind: str
    notes: Tuple[str, ...] = ()

    @property
    def word_count(self) -> int:
        return len(self.content.split())
