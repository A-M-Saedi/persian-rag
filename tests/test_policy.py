import pytest

from news_qa.policy import (
    DEFAULT_POLICY,
    GROUNDING_CLAUSE,
    OPEN,
    OUTSIDE_CONTEXT_MARKER,
    POLICIES,
    STRICT,
    build_prompt,
    resolve,
)
from news_qa.rag import PROMPT_TEMPLATE

QUESTION = 'سرمربی جدید تراکتور کیست؟'
SENTINEL = 'جواد نکونام سرمربی تراکتور شد.'


# ---------------------------------------------------------------- گراندبودن

def test_strict_forbids_outside_knowledge_and_open_does_not():
    assert GROUNDING_CLAUSE in STRICT.prompt_template
    assert GROUNDING_CLAUSE not in OPEN.prompt_template
    assert GROUNDING_CLAUSE not in OPEN.unsupported_template


def test_strict_refuses_without_context_and_open_does_not():
    assert build_prompt(STRICT, [], QUESTION) is None
    assert build_prompt(OPEN, [], QUESTION) is not None


def test_the_open_prompt_demands_a_provenance_marker():
    assert OUTSIDE_CONTEXT_MARKER in OPEN.prompt_template
    assert OUTSIDE_CONTEXT_MARKER in OPEN.unsupported_template


def test_every_policy_embeds_the_retrieved_context_verbatim():
    for name, policy in POLICIES.items():
        prompt = build_prompt(policy, [SENTINEL], QUESTION)
        assert SENTINEL in prompt, name
        assert QUESTION in prompt, name


def test_the_strict_prompt_is_byte_identical_to_the_old_one():
    chunks = [SENTINEL, 'تراکتور در تبریز بازی می‌کند.']
    assert build_prompt(STRICT, chunks, QUESTION) == PROMPT_TEMPLATE.format(
        context="\n\n".join(chunks), question=QUESTION
    )


def test_the_question_reaches_the_model_unnormalized():
    arabic_keyboard = 'سرمربي جديد تراكتور كيست؟'
    assert arabic_keyboard in build_prompt(STRICT, [SENTINEL], arabic_keyboard)


# ------------------------------------------------------------ رجیستری و پیش‌فرض

def test_the_default_policy_is_strict():
    assert DEFAULT_POLICY is STRICT
    assert DEFAULT_POLICY.name == 'strict'


def test_the_registry_key_matches_the_policy_name():
    for key, policy in POLICIES.items():
        assert policy.name == key
        assert resolve(key) is policy


def test_an_unknown_policy_name_is_rejected():
    for bad in ['', 'OPEN', 'open ', ' strict', 'چرت', 'Strict', None]:
        with pytest.raises((KeyError, TypeError)):
            resolve(bad)


def test_the_two_policies_are_actually_two():
    assert STRICT.name != OPEN.name
    assert STRICT.label != OPEN.label
    assert STRICT.prompt_template != OPEN.prompt_template
