import requests
from bs4 import BeautifulSoup

from .document import SourceError
from .normalize import normalize

CONTENT_SELECTORS = ['article', '.content', '.post-content', '.entry-content', '.news-body']

MIN_PARAGRAPH_CHARS = 20

THIN_ARTICLE_WORDS = 120

NOISE_TAGS = ['script', 'style', 'nav', 'header', 'footer', 'aside']

DEFAULT_TITLE = "بدون عنوان"


class ScrapeError(SourceError):
    pass


class PersianNewsScraper:
    def __init__(self, timeout=15):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.timeout = timeout

    @staticmethod
    def _para_text(p):
        return p.get_text(' ', strip=True)

    def clean_text(self, text):
        return normalize(text)

    def _paragraphs_to_text(self, paragraphs):
        texts = (self._para_text(p) for p in paragraphs)
        return '\n\n'.join(t for t in texts if len(t) > MIN_PARAGRAPH_CHARS)

    def _extract_title(self, soup):
        og_title = soup.find('meta', attrs={'property': 'og:title'})
        if og_title:
            text = self.clean_text(og_title.get('content') or '')
            if text:
                return text

        if soup.title:
            text = self.clean_text(soup.title.get_text(' ', strip=True))
            if text:
                return text

        return DEFAULT_TITLE

    def extract_from_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')

        title = self._extract_title(soup)

        for tag in soup(NOISE_TAGS):
            tag.decompose()

        content = ""
        for selector in CONTENT_SELECTORS:
            content_tag = soup.select_one(selector)
            if content_tag:
                content = self._paragraphs_to_text(content_tag.find_all('p'))
                if content:
                    break

        if not content:
            content = self._paragraphs_to_text(soup.find_all('p'))

        return {'title': title, 'content': self.clean_text(content)}

    def fetch(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ScrapeError(str(e)) from e

        response.encoding = 'utf-8'
        return response.text

    def extract_article(self, url):
        return self.extract_from_html(self.fetch(url))
