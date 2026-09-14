import math
import re
from collections import Counter

from .normalize import normalize

_WORD = re.compile(r'[\w\u200c]+')

STOPWORDS = frozenset("""
که از این را با برای است در به های ها یک هم می شد شده بود اما یا تا بر آن
هر چه چند کدام کجا کی چرا چگونه چیست کیست چیزی کسی آیا
خود دیگر همه نیز باید شود کرد کند کنند دارد دارند داشت داشته
اگر ولی پس چون وقتی حال طور مورد بین روی زیر بالا کنار
من تو او ما شما ایشان خویش
""".split())

K1 = 1.5
B = 0.75


def tokenize(text):
    return [
        token
        for token in _WORD.findall(normalize(text).lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


class BM25:
    def __init__(self, documents, k1=K1, b=B):
        self.k1 = k1
        self.b = b
        self.token_lists = [tokenize(document) for document in documents]
        self.lengths = [len(tokens) for tokens in self.token_lists]
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.frequencies = [Counter(tokens) for tokens in self.token_lists]

        document_count = len(documents)
        containing = Counter()
        for tokens in self.token_lists:
            containing.update(set(tokens))

        self.idf = {
            token: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
            for token, count in containing.items()
        }

        self.max_idf = math.log(1 + (document_count + 0.5) / 0.5)

    def scores(self, query):
        query_tokens = [token for token in tokenize(query) if token in self.idf]
        if not query_tokens or not self.average_length:
            return [0.0] * len(self.token_lists)

        results = []
        for frequency, length in zip(self.frequencies, self.lengths):
            score = 0.0
            for token in query_tokens:
                count = frequency.get(token, 0)
                if not count:
                    continue
                norm = count + self.k1 * (1 - self.b + self.b * length / self.average_length)
                score += self.idf[token] * count * (self.k1 + 1) / norm
            results.append(score)
        return results

    def _weight(self, token):
        return self.idf.get(token, self.max_idf)

    def coverage(self, query):
        tokens = set(tokenize(query))
        total = sum(self._weight(token) for token in tokens)
        if not total:
            return [0.0] * len(self.token_lists)

        return [
            sum(self._weight(token) for token in tokens if frequency.get(token)) / total
            for frequency in self.frequencies
        ]
