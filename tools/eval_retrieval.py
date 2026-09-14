#!/usr/bin/env python
"""گزارش بازیابی روی مجموعهٔ ارزیابی — عددی که هر مرحلهٔ بعدی باید از آن بهتر شود.

    python tools/eval_retrieval.py
    python tools/eval_retrieval.py --threshold 0.70 --pool 5
    python tools/eval_retrieval.py --evalset live

هیچ چیزی را تغییر نمی‌دهد و به ollama نیاز ندارد؛ فقط می‌سنجد. برای مقایسهٔ دو
تنظیم، دو بار با پارامترهای متفاوت اجرا کنید.

تفاوت «gate» و «rank» در ستون علت مهم است و همان چیزی است که پیش از این نما
دیده نمی‌شد:
    gate — چانکِ درست بین نامزدها بود، ولی فاصله‌اش از آستانه بیشتر شد.
           درمانش تنظیم آستانه است.
    rank — چانکِ درست اصلاً به عمقِ نامزد نرسید. آستانه هر چه باشد فرقی
           نمی‌کند؛ درمانش بازیابی بهتر است (کانال واژگانی، چانک‌بندی، --pool
           بزرگ‌تر). از آنجا که عمق و بودجه جدا شده‌اند، بزرگ کردنِ --pool دیگر
           لزوماً پرامپت را شلوغ‌تر نمی‌کند.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from news_qa import PersianNewsScraper, PersianRAG
from news_qa.rag import (
    CANDIDATE_POOL,
    DENSE_COVERAGE_FLOOR,
    LEXICAL_COVERAGE_FLOOR,
    PROMPT_BUDGET,
    RELEVANCE_THRESHOLD,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / 'tests' / 'fixtures'

EVALSETS = {'default': 'evalset.json', 'live': 'evalset_live.json'}


def load(name):
    with open(FIXTURE_DIR / name, encoding='utf-8') as f:
        return json.load(f)


def build_index(rag, manifest, fixtures):
    scraper = PersianNewsScraper()
    for name in fixtures:
        with open(FIXTURE_DIR / 'html' / f'{name}.html', encoding='utf-8') as f:
            article = scraper.extract_from_html(f.read())
        rag.chunk_and_store(
            article['content'], {'title': article['title'], 'url': manifest[name]['url']}
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold', type=float, default=RELEVANCE_THRESHOLD)
    parser.add_argument('--pool', type=int, default=CANDIDATE_POOL)
    parser.add_argument('--budget', type=int, default=PROMPT_BUDGET)
    parser.add_argument('--coverage', type=float, default=LEXICAL_COVERAGE_FLOOR)
    parser.add_argument('--dense-coverage', type=float, default=DENSE_COVERAGE_FLOOR)
    parser.add_argument('--evalset', choices=sorted(EVALSETS), default='default')
    args = parser.parse_args()

    manifest, evalset = load('manifest.json'), load(EVALSETS[args.evalset])
    fixtures = sorted({row['fixture'] for row in evalset['answerable']})

    rag = PersianRAG(
        persist_directory=None,
        relevance_threshold=args.threshold,
        candidate_pool=args.pool,
        prompt_budget=args.budget,
        lexical_coverage=args.coverage,
        dense_coverage=args.dense_coverage,
    )
    build_index(rag, manifest, fixtures)

    misses = []
    hits, trimmed = 0, []
    for row in evalset['answerable']:
        rag.current_url = manifest[row['fixture']]['url']
        candidates = rag.retrieve(row['question'])
        if any(c.kept and row['answer_key'] in c.document for c in candidates):
            hits += 1
            if not any(c.in_prompt and row['answer_key'] in c.document for c in candidates):
                trimmed.append(row)
            continue
        reachable = any(row['answer_key'] in c.document for c in candidates)
        nearest = min((c.distance for c in candidates), default=float('nan'))
        misses.append((row, 'gate' if reachable else 'rank', nearest))

    admitted = [
        (name, question)
        for question in evalset['unrelated']
        for name in fixtures
        if any(c.kept for c in _scoped(rag, manifest, name, question))
    ]

    total = len(evalset['answerable'])
    unrelated_total = len(evalset['unrelated']) * len(fixtures)

    print(
        f"\nآستانه {args.threshold}  ·  عمقِ نامزد {args.pool}  ·  "
        f"بودجهٔ پرامپت {args.budget}  ·  "
        f"پوشش واژگانی {args.coverage}  ·  "
        f"پوشش برداری {args.dense_coverage}  ·  {len(fixtures)} مقاله"
    )
    print(f"مجموعه {args.evalset} — {EVALSETS[args.evalset]}")
    print("=" * 64)
    print(f"چانکِ حاوی پاسخ بازیابی شد : {hits:>3} از {total:<3} = {hits / total:.0%}")
    if trimmed:
        print(f"  از این تعداد، بودجهٔ پرامپت برید : {len(trimmed):>3}  (به مدل نرسید)")
    print(
        f"سوال بی‌ربط متن گرفت       : {len(admitted):>3} از {unrelated_total:<3} = "
        f"{len(admitted) / unrelated_total:.0%}"
    )

    if misses:
        print(f"\nناکامی‌ها ({len(misses)}):")
        for row, cause, nearest in sorted(misses, key=lambda m: m[1]):
            print(f"  [{cause}] {nearest:.3f}  {row['question']}")
            print(f"           انتظار: {row['answer_key']}  ({row['fixture']})")
        by_gate = sum(1 for m in misses if m[1] == 'gate')
        print(
            f"\n  {by_gate} مورد با تنظیم آستانه قابل حل است، "
            f"{len(misses) - by_gate} مورد نیازمند بازیابی بهتر."
        )

    if admitted:
        print(f"\nپذیرفته‌شده‌های نادرست ({len(admitted)}):")
        for name, question in admitted:
            print(f"  {name}  ←  {question}")

    return 0


def _scoped(rag, manifest, name, question):
    rag.current_url = manifest[name]['url']
    return rag.retrieve(question)


if __name__ == '__main__':
    sys.exit(main())
