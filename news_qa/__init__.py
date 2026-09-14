import os

# Chroma's usage ping is opt-out and fires before any client call. Set it
# here, ahead of the chromadb import, so the system stays fully offline.
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')

from .normalize import normalize
from .policy import DEFAULT_POLICY, POLICIES, resolve
from .document import Document, SourceError
from .scraper import PersianNewsScraper, ScrapeError
from .sources import load_file, load_url, supported_suffixes
from .rag import (
    CANDIDATE_POOL,
    GenerationError,
    PersianRAG,
    PROMPT_BUDGET,
    RELEVANCE_THRESHOLD,
    Retrieved,
)

__all__ = [
    'CANDIDATE_POOL',
    'DEFAULT_POLICY',
    'Document',
    'GenerationError',
    'PersianNewsScraper',
    'PersianRAG',
    'POLICIES',
    'PROMPT_BUDGET',
    'RELEVANCE_THRESHOLD',
    'Retrieved',
    'ScrapeError',
    'SourceError',
    'load_file',
    'load_url',
    'normalize',
    'resolve',
    'supported_suffixes',
]
