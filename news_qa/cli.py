from .normalize import normalize
from .policy import DEFAULT_POLICY, OPEN, STRICT
from .rag import PersianRAG
from .rag import GenerationError
from .sources import (
    KIND_WEB, SourceError, is_thin, load_file, load_url, supported_suffixes,
)

EXIT_WORDS = ['خروج', 'exit', 'quit']
DEBUG_WORDS = ['دیباگ', 'debug']
HELP_WORDS = ['راهنما', 'help', '?', '؟']
MODE_WORDS = ['حالت', 'mode']

STRICT_WORDS = ['1', 'strict', 'سخت‌گیرانه', 'سختگیرانه']
OPEN_WORDS = ['2', 'open', 'آزاد', 'باز']

FILE_WORDS = ['1', 'file', 'فایل', 'فایل محلی']
URL_WORDS = ['2', 'url', 'وب', 'سایت', 'آدرس', 'لینک']

YES_WORDS = ['y', 'yes', 'بله', 'آری', 'آره', 'اره']

PREVIEW_CHARS = 70

EXTRACT_PREVIEW_CHARS = 240

HELP_TEXT = """دستورها:
  خروج / exit    پایان گفتگو
  دیباگ / debug  روشن و خاموش کردن نمای بازیابی
  حالت / mode    عوض کردن سیاست پاسخ (فقط متن خبر / متن خبر + دانش مدل)
  راهنما / help  همین متن
هر چیز دیگری به‌عنوان سوال از خبر پرسیده می‌شود."""


MODE_PROMPT = """
🔐 سیاست پاسخ:
  [۱] فقط متن خبر    — اگر پاسخ در خبر نباشد، امتناع می‌کند.
  [۲] متن خبر + دانش مدل — خبر مقدم است، ولی جای خالی را از دانش خودش پر می‌کند."""


SOURCE_PROMPT = """
📥 منبع شما کدام است؟
  [۱] فایل محلی  — {suffixes}
  [۲] آدرس وب    — صفحهٔ یک خبر"""


CHANNEL_WIDTH = 13


def resolve_policy_choice(answer, current):
    choice = normalize(answer).strip().lower()

    if not choice:
        return current
    if choice in STRICT_WORDS:
        return STRICT
    if choice in OPEN_WORDS:
        return OPEN
    return None


def is_affirmative(answer):
    return normalize(answer).strip().lower() in YES_WORDS


def format_extraction(document):
    words = document.word_count
    flat = ' '.join(document.content.split())
    preview = flat[:EXTRACT_PREVIEW_CHARS] + ("…" if len(flat) > EXTRACT_PREVIEW_CHARS else "")

    lines = [
        f"📌 عنوان: {document.title}",
        f"📄 تعداد کلمات: {words}",
        f"📝 {preview}",
    ]
    if is_thin(document):
        if document.kind == KIND_WEB:
            lines.append(
                f"⚠️  فقط {words} کلمه استخراج شد. اگر متنِ بالا خبری که می‌خواستید "
                f"نیست، احتمالاً استخراج به‌جای متنِ خبر بخشِ دیگری از صفحه را "
                f"برداشته است."
            )
        else:
            lines.append(
                f"⚠️  فقط {words} کلمه استخراج شد. اگر متنِ بالا با آنچه در فایل "
                f"می‌بینید نمی‌خواند، احتمالاً استخراج به متنِ اصلی نرسیده است."
            )
    lines.extend(f"ℹ️  {note}" for note in document.notes)
    return "\n".join(lines)


def format_retrieval(candidates, threshold, policy=None, budget=None):
    if not candidates:
        return "🔎 بازیابی: هیچ نامزدی برنگشت (مقاله‌ای نمایه نشده یا فیلتر url خالی است)."

    lines = [f"🔎 بازیابی — آستانه {threshold:.2f}:"]
    for i, c in enumerate(candidates, 1):
        if not c.kept:
            mark = "⛔️ رد   " if c.distance > threshold else "⛔️ پوشش "
        elif not c.in_prompt:
            mark = "✂️ بودجه"
        else:
            mark = "✅ قبول "
        channel = {
            'lexical': f"واژگانی {c.lexical_score:.2f}",
            'both': f"هر دو {c.lexical_score:.2f}",
        }.get(c.source, "برداری").ljust(CHANNEL_WIDTH)
        flat = c.document.replace("\n", " ")
        preview = flat[:PREVIEW_CHARS] + ("…" if len(flat) > PREVIEW_CHARS else "")
        lines.append(f"  {i}. {mark}  فاصله {c.distance:.3f}  {channel}  {preview}")

    trimmed = [c for c in candidates if c.kept and not c.in_prompt]
    if trimmed:
        lines.append(
            f"  ↳ {len(trimmed)} نامزد از دروازه رد شد ولی بودجهٔ پرامپت "
            f"({budget}) به مدل نرساندش."
        )

    untraced = [c for c in candidates if not c.kept and c.distance <= threshold]
    if untraced:
        lines.append(
            f"  ↳ {len(untraced)} نامزد از آستانه نزدیک‌تر بود ولی هیچ واژهٔ "
            f"محتوایی با سوال مشترک نداشت (هم‌نامی)."
        )

    blocked = [c for c in candidates if not c.kept]
    if blocked and not any(c.kept for c in candidates):
        far = [c for c in blocked if c.distance > threshold]
        if far:
            lines.append(
                f"  ↳ نزدیک‌ترین نامزد {far[0].distance:.3f} بود، یعنی "
                f"{far[0].distance - threshold:.3f} دورتر از آستانه."
            )
        if policy is not None and policy.unsupported_template is not None:
            lines.append(
                f"  ↳ سیاست «{policy.label}»: پاسخ بدون هیچ متنی از خبر ساخته شد."
            )
    return "\n".join(lines)


def resolve_source_choice(answer):
    choice = normalize(answer).strip().lower()
    if choice in FILE_WORDS:
        return 'file'
    if choice in URL_WORDS:
        return 'url'
    return None


def _ask_for_source():
    print(SOURCE_PROMPT.format(suffixes='، '.join(supported_suffixes())))
    while True:
        answer = input("👉 انتخاب کنید [۱/۲]: ").strip()
        if answer.lower() in EXIT_WORDS:
            return None
        kind = resolve_source_choice(answer)
        if kind is None:
            print("⚠️ نفهمیدم. ۱ برای فایل، ۲ برای آدرس وب.")
            continue

        if kind == 'file':
            raw = input("\n📁 مسیر فایل را وارد کنید: ").strip().strip('"\'')
            if not raw:
                print("❌ مسیری وارد نشد.")
                return None
            print("\n⏳ در حال خواندن و پردازش فایل...")
            return load_file(raw)

        url = input("\n🌐 آدرس کامل خبر را وارد کنید: ").strip()
        if not url:
            print("❌ آدرسی وارد نشد.")
            return None
        print("\n⏳ در حال دریافت و پردازش خبر...")
        return load_url(url)


def _ask_for_policy(current):
    print(MODE_PROMPT)
    while True:
        answer = input(f"👉 انتخاب کنید (فعلی: {current.label}): ").strip()
        if answer.lower() in EXIT_WORDS:
            return None

        policy = resolve_policy_choice(answer, current)
        if policy is None:
            print("⚠️ نفهمیدم. ۱ یا ۲ بزنید (یا Enter برای نگه داشتنِ حالت فعلی).")
            continue

        print(f"🔐 سیاست پاسخ: {policy.label}")
        return policy


def _confirm_ungrounded():
    print("\n⚠️ هیچ بخش مرتبطی در متن خبر پیدا نشد.")
    return is_affirmative(
        input("   پاسخ از دانش عمومی مدل ساخته شود؟ (بله/خیر): ")
    )


def main():
    print("=" * 60)
    print("🔍 سیستم پرسش و پاسخ از خبر با RAG")
    print("=" * 60)

    try:
        document = _ask_for_source()
    except SourceError as e:
        print(f"❌ دریافت منبع شکست خورد: {e}")
        return 1

    if document is None:
        return 1

    if not document.content:
        print("❌ متنی از این منبع استخراج نشد.")
        for note in document.notes:
            print(f"   {note}")
        return 1

    print("\n✅ منبع با موفقیت استخراج شد.")
    print(format_extraction(document))
    print("-" * 60)

    print("\n⏳ در حال راه‌اندازی سیستم RAG...")
    try:
        rag = PersianRAG()

        print("\n⏳ در حال پردازش و ذخیره‌سازی خبر...")
        rag.chunk_and_store(document.content, {
            'title': document.title,
            'url': document.source_id,
            'kind': document.kind,
        })
        print("✅ منبع در پایگاه داده ذخیره شد.\n")

    except Exception as e:
        print(f"❌ خطا در راه‌اندازی RAG: {e}")
        return 1

    policy = _ask_for_policy(DEFAULT_POLICY)
    if policy is None:
        print("👋 خداحافظ!")
        return 0

    print("\n💬 حالا سوال خود را بپرسید. (راهنما: 'راهنما' — خروج: 'خروج')")
    debug = False
    while True:
        question = input("\n❓ سوال شما: ").strip()
        command = question.lower()

        if command in EXIT_WORDS:
            print("👋 خداحافظ!")
            break
        if not question:
            print("⚠️ لطفاً سوالی وارد کنید.")
            continue
        if command in HELP_WORDS:
            print(HELP_TEXT)
            continue
        if command in DEBUG_WORDS:
            debug = not debug
            print("🔎 نمای بازیابی روشن شد." if debug else "🔎 نمای بازیابی خاموش شد.")
            continue
        if command in MODE_WORDS:
            chosen = _ask_for_policy(policy)
            if chosen is None:
                print("👋 خداحافظ!")
                break
            policy = chosen
            continue

        print("\n⏳ در حال پردازش...")

        chunks = rag.retrieve_relevant_chunks(question)
        if not chunks and policy.unsupported_template is not None:
            if not _confirm_ungrounded():
                print("   باشد — پاسخی ساخته نشد.")
                continue

        try:
            answer = rag.answer_from(question, chunks, policy)
        except GenerationError as e:
            print(f"\n❌ مدل زبانی در دسترس نیست: {e}")
            print("   مطمئن شوید `ollama serve` در حال اجراست، بعد دوباره بپرسید.")
            continue

        if debug:
            print()
            print(format_retrieval(
                rag.last_retrieval, rag.relevance_threshold, policy, rag.prompt_budget
            ))

        print("\n" + "=" * 60)
        print(f"🤖 پاسخ مدل — سیاست: {policy.label}")
        print("=" * 60)
        print(answer)
        print("=" * 60)

    return 0
