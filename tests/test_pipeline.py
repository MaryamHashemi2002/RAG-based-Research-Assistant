from src.generation.pipeline import EMPTY_MESSAGE, ask, clean_answer
from src.generation.prompt import NO_ANSWER

CHUNKS = [
    {"title": "Paper A", "page": 3, "year": 2020, "paper_id": "a1", "text": "alpha"},
    {"title": "Paper B", "page": 7, "year": 2021, "paper_id": "b2", "text": "beta"},
]


def retrieve(question, k, **filters):
    return CHUNKS[:k]


def test_normal_answer_returns_text_and_numbered_sources():
    result = ask("q?", retrieve=retrieve, llm=lambda m: "  An answer [1].  ")
    assert result.status == "ok"
    assert result.text == "An answer [1]."
    assert [s["n"] for s in result.sources] == [1, 2]
    assert result.sources[0]["title"] == "Paper A"


def test_refusal_drops_extra_text_and_sources():
    raw = f"Some invented details. {NO_ANSWER}"
    result = ask("q?", retrieve=retrieve, llm=lambda m: raw)
    assert result.status == "no_answer"
    assert result.text == NO_ANSWER
    assert result.sources == []


def test_empty_model_answer_is_reported():
    result = ask("q?", retrieve=retrieve, llm=lambda m: "   ")
    assert result.status == "empty_answer"
    assert result.text == EMPTY_MESSAGE


def test_no_retrieved_chunks_skips_the_model():
    def llm_must_not_run(messages):
        raise AssertionError("model should not be called")

    result = ask("q?", retrieve=lambda q, k, **f: [], llm=llm_must_not_run)
    assert result.status == "no_results"
    assert result.text == NO_ANSWER


def test_blank_question_is_rejected():
    result = ask("   ", retrieve=retrieve, llm=lambda m: "x")
    assert result.status == "empty_question"


def test_model_errors_do_not_crash():
    def broken(messages):
        raise ConnectionError("no server")

    result = ask("q?", retrieve=retrieve, llm=broken)
    assert result.status == "llm_error"
    assert "Ollama" in result.text


def test_clean_answer_is_case_insensitive_about_the_refusal():
    text, status = clean_answer(NO_ANSWER.upper())
    assert status == "no_answer"