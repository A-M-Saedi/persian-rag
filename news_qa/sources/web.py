from __future__ import annotations

from ..document import KIND_WEB, Document
from ..scraper import PersianNewsScraper


def load_url(url: str, scraper: PersianNewsScraper = None) -> Document:
    scraper = scraper or PersianNewsScraper()
    article = scraper.extract_article(url)
    return Document(
        title=article['title'],
        content=article['content'],
        source_id=url,
        kind=KIND_WEB,
    )
