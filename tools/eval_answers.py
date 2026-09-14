#!/usr/bin/env python
"""گزارش پاسخِ سرتاسری روی مجموعهٔ ارزیابی — مکملِ eval_retrieval.py.

    python tools/eval_answers.py
    python tools/eval_answers.py --threshold 0.60
    python tools/eval_answers.py --policy open
    python tools/eval_answers.py --evalset live
    python tools/eval_answers.py --pool 8 --budget 8

برخلاف eval_retrieval.py این یکی به ollama نیاز دارد و کند است (هر سوال یک
فراخوانی مدل). دلیل جدا بودنشان همین است: سنجشِ بازیابی باید ارزان و آفلاین
بماند تا در هر تغییری اجرا شود.

چیزی که این ابزار نشان می‌دهد و آزمونِ endtoend نمی‌دهد، تفکیکِ علتِ خطاست:
    بازیابی — متنِ درست اصلاً به مدل نرسید. تقصیر بازیابی است. «نرسیدن»
              یعنی in_prompt، نه kept: چانکی که از دروازه رد شد ولی بودجهٔ
              پرامپت بریدش هم به مدل نرسیده، و باید در همین سطل بیفتد.
    تولید   — متنِ درست رسید ولی پاسخ غلط شد. تقصیر مدل یا پرامپت است.

بدون این تفکیک، «۱۷ از ۲۸» نمی‌گوید کدام لایه باید درست شود.

عددِ منتشرشده در README زیر --policy strict و --evalset default سنجیده شده و
باید همان‌طور بماند. دلیلِ اولی در «پاسخ بی‌پشتوانه» پایین‌تر آمده است؛ دومی
مجموعهٔ کالیبره‌شده است و مجموعهٔ live عمداً عددِ دیگری می‌دهد.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from news_qa import GenerationError, POLICIES, PersianRAG, resolve
from news_qa.policy import OUTSIDE_CONTEXT_MARKER
from news_qa.rag import (
    CANDIDATE_POOL,
    DENSE_COVERAGE_FLOOR,
    LEXICAL_COVERAGE_FLOOR,
    NO_CONTEXT_MESSAGE,
    PROMPT_BUDGET,
    RELEVANCE_THRESHOLD,
)
from tools.eval_retrieval import EVALSETS, build_index, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold', type=float, default=RELEVANCE_THRESHOLD)
    parser.add_argument('--pool', type=int, default=CANDIDATE_POOL)
    parser.add_argument('--budget', type=int, default=PROMPT_BUDGET)
    parser.add_argument('--coverage', type=float, default=LEXICAL_COVERAGE_FLOOR)
    parser.add_argument('--dense-coverage', type=float, default=DENSE_COVERAGE_FLOOR)
    parser.add_argument('--policy', choices=sorted(POLICIES), default='strict')
    parser.add_argument('--evalset', choices=sorted(EVALSETS), default='default')
    args = parser.parse_args()
    policy = resolve(args.policy)

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

    correct, ungrounded, retrieval_faults, generation_faults = 0, [], [], []
    for index, row in enumerate(evalset['answerable']):
        rag.current_url = manifest[row['fixture']]['url']
        try:
            answer = rag.ask(row['question'], policy)
        except GenerationError as e:
            print(f"\n❌ مدل زبانی در دسترس نیست: {e}")
            print(f"   اجرا در سوال {index + 1} از {len(evalset['answerable'])} متوقف شد؛")
            print("   عددی چاپ نمی‌شود. `ollama serve` را بالا بیاورید و دوباره اجرا کنید.")
            return 2
        reached = any(
            c.in_prompt and row['answer_key'] in c.document for c in rag.last_retrieval
        )
        if row['answer_key'] in answer:
            if OUTSIDE_CONTEXT_MARKER in answer:
                ungrounded.append((row, answer))
            else:
                correct += 1
        elif reached:
            generation_faults.append((row, answer))
        else:
            retrieval_faults.append((row, answer))

    total = len(evalset['answerable'])
    reached_total = correct + len(generation_faults)

    print(
        f"\nآستانه {args.threshold}  ·  عمقِ نامزد {args.pool}  ·  "
        f"بودجهٔ پرامپت {args.budget}  ·  "
        f"پوشش واژگانی {args.coverage}  ·  "
        f"پوشش برداری {args.dense_coverage}  ·  {len(fixtures)} مقاله"
    )
    print(f"مجموعه {args.evalset} — {EVALSETS[args.evalset]}")
    print(f"سیاست {policy.name} — {policy.label}")
    print("=" * 64)
    print(f"پاسخ درست           : {correct:>3} از {total:<3} = {correct / total:.0%}")
    print(f"  ناکامیِ بازیابی   : {len(retrieval_faults):>3}  (متن به مدل نرسید)")
    print(f"  ناکامیِ تولید     : {len(generation_faults):>3}  (متن رسید، پاسخ غلط شد)")
    print(f"  پاسخ بی‌پشتوانه   : {len(ungrounded):>3}  (درست بود، ولی از خبر نیامد)")
    if reached_total:
        print(
            f"\nوقتی متنِ درست رسید: {correct} از {reached_total} = "
            f"{correct / reached_total:.0%} درست"
        )

    for title, faults in [
        ("ناکامیِ بازیابی", retrieval_faults),
        ("ناکامیِ تولید", generation_faults),
        ("پاسخ بی‌پشتوانه", ungrounded),
    ]:
        if not faults:
            continue
        print(f"\n{title} ({len(faults)}):")
        for row, answer in faults:
            shown = "«پیدا نشد»" if answer == NO_CONTEXT_MESSAGE else answer.replace("\n", " ")[:70]
            print(f"  {row['question']}")
            print(f"    انتظار: {row['answer_key']}")
            print(f"    پاسخ:  {shown}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
