from src.generation.prompt import MAX_CHARS, NO_ANSWER, SYSTEM_PROMPT, build_messages, format_sources


def chunks():
    return [
        {"title": "Paper A", "page": 3, "text": "alpha text"},
        {"title": "Paper B", "page": 7, "text": "beta text"},
    ]


def test_sources_are_numbered_from_one_with_title_and_page():
    out = format_sources(chunks())
    assert out.startswith("[1] Paper A (p.3)\nalpha text")
    assert "[2] Paper B (p.7)\nbeta text" in out


def test_long_sources_are_truncated():
    long_chunk = [{"title": "T", "page": 1, "text": "x" * (MAX_CHARS + 500)}]
    assert format_sources(long_chunk).count("x") == MAX_CHARS


def test_messages_have_system_then_user_with_question():
    messages = build_messages("What is lasso?", chunks())
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Question: What is lasso?" in messages[1]["content"]


def test_system_prompt_contains_refusal_sentence():
    assert NO_ANSWER in SYSTEM_PROMPT