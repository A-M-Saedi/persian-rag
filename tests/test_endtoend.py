import pytest

from news_qa.policy import OPEN, OUTSIDE_CONTEXT_MARKER, REFUSAL_SENTENCE, STRICT
from news_qa.rag import NO_CONTEXT_MESSAGE

pytestmark = pytest.mark.endtoend

MIN_CORRECT_ANSWERS = 21


@pytest.fixture(scope='module')
def answers(indexed_rag, manifest, evalset):
    collected = []
    for row in evalset['answerable']:
        indexed_rag.current_url = manifest[row['fixture']]['url']
        answer = indexed_rag.ask(row['question'], STRICT)
        collected.append((row, answer, list(indexed_rag.last_retrieval)))
    return collected


def test_answer_accuracy_meets_the_measured_budget(answers):
    wrong = [
        (row['question'], row['answer_key'], answer)
        for row, answer, _ in answers
        if row['answer_key'] not in answer
    ]
    correct = len(answers) - len(wrong)
    detail = "\n".join(f"  {q}\n    انتظار: {k}\n    پاسخ:  {a}" for q, k, a in wrong)
    assert correct >= MIN_CORRECT_ANSWERS, f"{correct} از {len(answers)}\n{detail}"


def test_the_model_never_answers_from_an_empty_context(answers):
    leaked = [
        (row['question'], answer)
        for row, answer, retrieval in answers
        if not any(r.kept for r in retrieval) and answer != NO_CONTEXT_MESSAGE
    ]
    assert not leaked, leaked


def test_unrelated_questions_are_refused(indexed_rag, manifest, evalset):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    for question in evalset['unrelated']:
        assert indexed_rag.ask(question, STRICT) == NO_CONTEXT_MESSAGE, question


def test_open_mode_answers_what_strict_mode_refuses(indexed_rag, manifest):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    question = 'پایتخت ژاپن کجاست؟'

    assert indexed_rag.ask(question, STRICT) == NO_CONTEXT_MESSAGE

    answer = indexed_rag.ask(question, OPEN)
    assert answer != NO_CONTEXT_MESSAGE, answer
    assert REFUSAL_SENTENCE not in answer, answer
    assert not answer.startswith('❌'), answer


MIN_MARKED_UNRELATED = 4


def test_open_mode_marks_most_of_what_did_not_come_from_the_article(
    indexed_rag, manifest, evalset
):
    indexed_rag.current_url = manifest['varzesh3_2410970']['url']
    answers = [indexed_rag.ask(q, OPEN) for q in evalset['unrelated']]
    marked = [a for a in answers if OUTSIDE_CONTEXT_MARKER in a]

    detail = "\n".join(f"  {q}\n    {a[:80]}" for q, a in zip(evalset['unrelated'], answers))
    assert len(marked) >= MIN_MARKED_UNRELATED, f"{len(marked)} از {len(answers)}\n{detail}"
