from collections import namedtuple

GROUNDING_CLAUSE = 'از دانش عمومی خودت استفاده نکن'

REFUSAL_SENTENCE = 'در متن خبر پاسخی برای این سوال وجود ندارد'

FROM_CONTEXT_MARKER = 'طبق متن خبر:'
OUTSIDE_CONTEXT_MARKER = 'خارج از متن خبر:'


GenerationPolicy = namedtuple(
    'GenerationPolicy',
    ['name', 'label', 'prompt_template', 'unsupported_template'],
)


STRICT_PROMPT = """شما یک دستیار پرسش و پاسخ فارسی هستید.

قوانین:
- فقط و فقط بر اساس «متن» زیر پاسخ بده.
- اگر پاسخ در متن نیامده است، دقیقاً این جمله را بنویس: در متن خبر پاسخی برای این سوال وجود ندارد.
- از دانش عمومی خودت استفاده نکن و چیزی به متن اضافه نکن.
- پاسخ را کوتاه و فقط به زبان فارسی بنویس.

متن:
----------------------------------------
{context}
----------------------------------------

سوال: {question}

پاسخ به زبان فارسی:"""


OPEN_PROMPT = """شما یک دستیار پرسش و پاسخ فارسی هستید.

قوانین:
- اول «متن» زیر را بخوان. هر چه پاسخش در متن هست، از متن بردار؛ متن خبر مقدم است.
- اگر متن پاسخ را کامل نداشت، می‌توانی از دانش عمومی خودت کمک بگیری.
- منبع را مشخص کن: جمله‌ای که از متن می‌آید با «طبق متن خبر:» و جمله‌ای که از دانش عمومی می‌آید با «خارج از متن خبر:» شروع شود.
- چیزی نگو که با متن خبر در تضاد باشد. اگر دانش عمومی‌ات با متن نمی‌خواند، متن را بنویس.
- اگر مطمئن نیستی، بگو مطمئن نیستی؛ از خودت چیزی نساز.
- پاسخ را کوتاه و فقط به زبان فارسی بنویس.

متن:
----------------------------------------
{context}
----------------------------------------

سوال: {question}

پاسخ به زبان فارسی:"""


OPEN_UNSUPPORTED_PROMPT = """شما یک دستیار پرسش و پاسخ فارسی هستید.

هیچ بخش مرتبطی در متن خبر پیدا نشد. کاربر صریحاً اجازه داده است که در این حالت
از دانش عمومی خودت پاسخ بدهی.

قوانین:
- پاسخ را با «خارج از متن خبر:» شروع کن، چون این پاسخ از خبر نمی‌آید.
- اگر مطمئن نیستی، بگو مطمئن نیستی؛ از خودت چیزی نساز.
- پاسخ را کوتاه و فقط به زبان فارسی بنویس.

سوال: {question}

پاسخ به زبان فارسی:"""


STRICT = GenerationPolicy(
    name='strict',
    label='فقط متن خبر',
    prompt_template=STRICT_PROMPT,
    unsupported_template=None,
)

OPEN = GenerationPolicy(
    name='open',
    label='متن خبر + دانش مدل',
    prompt_template=OPEN_PROMPT,
    unsupported_template=OPEN_UNSUPPORTED_PROMPT,
)


POLICIES = {
    'strict': STRICT,
    'open': OPEN,
}

DEFAULT_POLICY = STRICT


def resolve(name):
    return POLICIES[name]


def build_prompt(policy, chunks, question):
    if not chunks:
        if policy.unsupported_template is None:
            return None
        template = policy.unsupported_template
    else:
        template = policy.prompt_template

    return template.format(context="\n\n".join(chunks), question=question)
