import hashlib
from collections import namedtuple

import chromadb
import numpy as np
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .lexical import BM25, tokenize
from .normalize import normalize
from .policy import DEFAULT_POLICY, STRICT_PROMPT, build_prompt
from .scraper import DEFAULT_TITLE

RELEVANCE_THRESHOLD = 0.63

CANDIDATE_POOL = 3

PROMPT_BUDGET = 6

MAX_LEXICAL_ADDITIONS = 3

LEXICAL_SCORE_FLOOR = 0.35

LEXICAL_COVERAGE_FLOOR = 0.35

DENSE_COVERAGE_FLOOR = 0.0

MAX_LEXICAL_CORPUS_CHUNKS = 2000

EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_PERSIST_DIR = './chroma_db'
COLLECTION_NAME = 'news_articles'

NO_CONTEXT_MESSAGE = "❌ هیچ بخش مرتبطی در متن خبر پیدا نشد."


class GenerationError(Exception):
    pass


PROMPT_TEMPLATE = STRICT_PROMPT


Retrieved = namedtuple(
    'Retrieved',
    ['document', 'distance', 'kept', 'source', 'lexical_score', 'in_prompt'],
)
Retrieved.__new__.__defaults__ = ('dense', 0.0, True)


def _cosine_distance(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if not norms:
        return 1.0
    return float(1.0 - np.dot(a, b) / norms)


def _apply_dense_coverage(candidates, question, index, floor):
    if index is None or floor < 0 or not set(tokenize(question)):
        return candidates

    documents, _, bm25 = index
    coverage = dict(zip(documents, bm25.coverage(question)))

    def traced(candidate):
        value = coverage.get(candidate.document)
        return value is None or value > floor

    return [
        c if traced(c) or not c.kept or c.source != 'dense' else c._replace(kept=False)
        for c in candidates
    ]


def _apply_budget(candidates, budget):
    dense = sorted(
        [c for c in candidates if c.kept and c.source != 'lexical'],
        key=lambda c: c.distance,
    )
    lexical = sorted(
        [c for c in candidates if c.kept and c.source == 'lexical'],
        key=lambda c: c.lexical_score,
        reverse=True,
    )

    chosen, order = set(), []
    while dense or lexical:
        if dense:
            order.append(dense.pop(0))
        if lexical:
            order.append(lexical.pop(0))
    for c in order[:budget]:
        chosen.add(c.document)

    return [c._replace(in_prompt=c.kept and c.document in chosen) for c in candidates]


class PersianRAG:
    def __init__(
        self,
        model_name='qwen3:4b-instruct',
        persist_directory=DEFAULT_PERSIST_DIR,
        relevance_threshold=RELEVANCE_THRESHOLD,
        candidate_pool=CANDIDATE_POOL,
        prompt_budget=PROMPT_BUDGET,
        lexical_coverage=LEXICAL_COVERAGE_FLOOR,
        dense_coverage=DENSE_COVERAGE_FLOOR,
    ):
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold
        self.candidate_pool = candidate_pool
        self.prompt_budget = prompt_budget
        self.lexical_coverage = lexical_coverage
        self.dense_coverage = dense_coverage
        self.last_thinking = ""
        self.last_retrieval = []
        self.last_policy = DEFAULT_POLICY
        self._lexical_cache = {}
        self.current_url = None

        print("⏳ در حال بارگذاری مدل امبدینگ...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("✅ مدل امبدینگ بارگذاری شد.")

        self.text_splitter = self._build_splitter()

        if persist_directory is None:
            self.chroma_client = chromadb.EphemeralClient()
        else:
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("✅ سیستم RAG آماده است.")

    def _build_splitter(self):
        tokenizer = self.embedding_model.tokenizer

        def token_len(text):
            return len(tokenizer.encode(text, add_special_tokens=False))

        return RecursiveCharacterTextSplitter(
            chunk_size=self.embedding_model.max_seq_length - 8,
            chunk_overlap=16,
            length_function=token_len,
            separators=["\n\n", "\n", ". ", "؟ ", "! ", "؛ ", "، ", " ", ""],
            keep_separator="end",
        )

    def chunk_and_store(self, text, metadata=None):
        text = normalize(text)

        chunks = self.text_splitter.split_text(text)

        title = normalize((metadata or {}).get('title') or '')
        if title and title != DEFAULT_TITLE and title not in chunks:
            chunks.insert(0, title)

        print(f"📄 متن به {len(chunks)} چانک تقسیم شد.")

        if not chunks:
            return False

        metadata = metadata or {}
        url = metadata.get('url')

        if url:
            self.collection.delete(where={'url': url})
            self.current_url = url

        self._lexical_cache.clear()

        ids = [
            hashlib.sha256(f"{url}|{i}|{chunk}".encode('utf-8')).hexdigest()
            for i, chunk in enumerate(chunks)
        ]

        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        metadatas = [dict(metadata) for _ in range(len(chunks))]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        return True

    def _lexical_index(self, url):
        if url in self._lexical_cache:
            return self._lexical_cache[url]

        stored = self.collection.get(
            where={'url': url} if url else None,
            include=['documents', 'embeddings'],
        )
        documents = stored['documents'] or []
        if not documents or len(documents) > MAX_LEXICAL_CORPUS_CHUNKS:
            self._lexical_cache[url] = None
            return None

        index = (documents, stored['embeddings'], BM25(documents))
        self._lexical_cache[url] = index
        return index

    def _lexical_endorsements(self, question, url):
        index = self._lexical_index(url)
        if index is None:
            return {}, index

        documents, _, bm25 = index
        scores = bm25.scores(question)
        coverage = bm25.coverage(question)
        best = max(scores) if scores else 0.0
        if best <= 0:
            return {}, index

        floor = best * LEXICAL_SCORE_FLOOR
        endorsed = {}
        for i in sorted(range(len(documents)), key=lambda i: scores[i], reverse=True):
            if len(endorsed) >= MAX_LEXICAL_ADDITIONS:
                break
            if scores[i] < floor:
                continue
            if coverage[i] < self.lexical_coverage:
                continue
            endorsed[documents[i]] = scores[i]
        return endorsed, index

    def retrieve(self, question, pool=None, budget=None, url='__current__'):
        if url == '__current__':
            url = self.current_url
        if pool is None:
            pool = self.candidate_pool
        if budget is None:
            budget = self.prompt_budget

        question_embedding = self.embedding_model.encode([normalize(question)])[0]
        results = self.collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=pool,
            where={'url': url} if url else None,
            include=['documents', 'distances']
        )

        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []

        candidates = [
            Retrieved(doc, dist, dist <= self.relevance_threshold, 'dense', 0.0)
            for doc, dist in zip(documents, distances)
        ]

        endorsed, index = self._lexical_endorsements(question, url)

        candidates = _apply_dense_coverage(
            candidates, question, index, self.dense_coverage
        )

        candidates = [
            c._replace(kept=True, source='both', lexical_score=endorsed[c.document])
            if c.document in endorsed and not c.kept else
            c._replace(source='both', lexical_score=endorsed[c.document])
            if c.document in endorsed else c
            for c in candidates
        ]

        seen = {c.document for c in candidates}
        documents_all, embeddings, _ = index if index else ([], [], None)
        position = {document: i for i, document in enumerate(documents_all)}
        candidates += [
            Retrieved(
                document=document,
                distance=_cosine_distance(question_embedding, embeddings[position[document]]),
                kept=True,
                source='lexical',
                lexical_score=score,
            )
            for document, score in endorsed.items()
            if document not in seen
        ]

        candidates = _apply_budget(candidates, budget)

        self.last_retrieval = sorted(candidates, key=lambda c: c.distance)
        return self.last_retrieval

    def retrieve_relevant_chunks(self, question, pool=None, budget=None, url='__current__'):
        return [
            r.document
            for r in self.retrieve(question, pool, budget, url)
            if r.in_prompt
        ]

    def answer_from(self, question, chunks, policy=DEFAULT_POLICY):
        self.last_policy = policy
        self.last_thinking = ""

        prompt = build_prompt(policy, chunks, question)

        if prompt is None:
            return NO_CONTEXT_MESSAGE

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.3,
                    'num_predict': 4096,
                }
            )
        except Exception as e:
            raise GenerationError(str(e)) from e

        message = response.message
        answer = (message.content or "").strip()
        self.last_thinking = (message.thinking or "").strip()

        if not answer:
            if response.done_reason == 'length':
                return "⚠️ بودجه تولید پیش از رسیدن به پاسخ تمام شد (num_predict را افزایش دهید)."
            return "⚠️ مدل پاسخی تولید نکرد. لطفاً دوباره امتحان کنید."

        return answer

    def ask(self, question, policy=DEFAULT_POLICY):
        return self.answer_from(
            question, self.retrieve_relevant_chunks(question), policy
        )
