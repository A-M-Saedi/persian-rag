import copy
import json
import os
import pathlib
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE_DIR = pathlib.Path(__file__).resolve().parent / 'fixtures'
HTML_DIR = FIXTURE_DIR / 'html'

RETRIEVAL_FIXTURES = ['varzesh3_2410970', 'mehr_6914285', 'yjc_9118143']

LIVE_FIXTURES = ['zoomit_465572', 'wikipedia_persian_gulf', 'hamshahri_1063549']

LIVE_COLLECTION = 'live_eval'


def _fixture_names():
    return sorted(p.stem for p in HTML_DIR.glob('*.html'))


@pytest.fixture(scope='session')
def manifest():
    with open(FIXTURE_DIR / 'manifest.json', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope='session')
def baseline():
    with open(FIXTURE_DIR / 'baseline_scraper.json', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope='session')
def evalset():
    with open(FIXTURE_DIR / 'evalset.json', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope='session')
def evalset_live():
    with open(FIXTURE_DIR / 'evalset_live.json', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope='session')
def load_html():
    def _load(name):
        return (HTML_DIR / f'{name}.html').read_bytes().decode('utf-8', errors='replace')
    return _load


@pytest.fixture(scope='session')
def scraper():
    from news_qa import PersianNewsScraper
    return PersianNewsScraper()


@pytest.fixture(scope='session')
def articles(scraper, load_html):
    return {name: scraper.extract_from_html(load_html(name)) for name in _fixture_names()}


@pytest.fixture(scope='session')
def rag():
    from news_qa import PersianRAG
    return PersianRAG(persist_directory=None)


@pytest.fixture(scope='session')
def indexed_rag(rag, articles, manifest):
    for name in RETRIEVAL_FIXTURES:
        rag.chunk_and_store(
            articles[name]['content'],
            {'title': articles[name]['title'], 'url': manifest[name]['url']},
        )
    return rag


@pytest.fixture(scope='session')
def indexed_rag_live(rag, articles, manifest):
    clone = copy.copy(rag)
    clone.collection = rag.chroma_client.get_or_create_collection(
        name=LIVE_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    clone._lexical_cache = {}
    for name in LIVE_FIXTURES:
        clone.chunk_and_store(
            articles[name]['content'],
            {'title': articles[name]['title'], 'url': manifest[name]['url']},
        )
    return clone


@pytest.fixture(scope='session')
def rag_factory(indexed_rag, manifest):
    def make(relevance_threshold=None, candidate_pool=None, prompt_budget=None,
             dense_coverage=None, fixture='varzesh3_2410970'):
        clone = copy.copy(indexed_rag)
        if relevance_threshold is not None:
            clone.relevance_threshold = relevance_threshold
        if candidate_pool is not None:
            clone.candidate_pool = candidate_pool
        if prompt_budget is not None:
            clone.prompt_budget = prompt_budget
        if dense_coverage is not None:
            clone.dense_coverage = dense_coverage
        clone.current_url = manifest[fixture]['url']
        return clone

    return make


def pytest_generate_tests(metafunc):
    if 'fixture_name' in metafunc.fixturenames:
        metafunc.parametrize('fixture_name', _fixture_names())
    if 'retrieval_fixture' in metafunc.fixturenames:
        metafunc.parametrize('retrieval_fixture', RETRIEVAL_FIXTURES)
