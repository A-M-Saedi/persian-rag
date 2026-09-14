import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hashlib

from news_qa import PersianNewsScraper

FIXTURE_DIR = ROOT / 'tests' / 'fixtures'
HTML_DIR = FIXTURE_DIR / 'html'
BASELINE = FIXTURE_DIR / 'baseline_scraper.json'

HEAD_CHARS = 160
TAIL_CHARS = 120


def build():
    scraper = PersianNewsScraper()
    out = {}
    for path in sorted(HTML_DIR.glob('*.html')):
        html = path.read_bytes().decode('utf-8', errors='replace')
        article = scraper.extract_from_html(html)
        content = article['content']
        out[path.stem] = {
            'title': article['title'],
            'content_chars': len(content),
            'content_words': len(content.split()),
            'content_sha256': hashlib.sha256(content.encode()).hexdigest(),
            'content_head': content[:HEAD_CHARS],
            'content_tail': content[-TAIL_CHARS:],
            'paragraphs': content.count('\n\n') + 1 if content else 0,
        }
    return out


def main():
    fresh = build()
    old = json.loads(BASELINE.read_text(encoding='utf-8')) if BASELINE.exists() else {}

    changed = False
    for name, new_row in fresh.items():
        old_row = old.get(name, {})
        for key in ('title', 'content_words', 'content_sha256'):
            if old_row.get(key) != new_row[key]:
                changed = True
                print(f"{name}.{key}")
                print(f"    - {old_row.get(key)!r}")
                print(f"    + {new_row[key]!r}")

    if not changed:
        print("بدون تغییر.")
        return 0

    if '--write' not in sys.argv:
        print("\n(برای نوشتن: python tools/regen_baseline.py --write)")
        return 1

    BASELINE.write_text(
        json.dumps(fresh, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(f"\nنوشته شد: {BASELINE.relative_to(ROOT)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
