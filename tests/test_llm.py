from types import SimpleNamespace

from src.generation.llm import ask


def fake_client(reply):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


def test_ask_returns_stripped_reply():
    client, _ = fake_client("  hello  \n")
    assert ask("hi", client=client) == "hello"


def test_ask_puts_system_message_first():
    client, calls = fake_client("ok")
    ask("question", system="be brief", client=client)
    messages = calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[1] == {"role": "user", "content": "question"}


def test_ask_without_system_sends_only_user_message():
    client, calls = fake_client("ok")
    ask("question", client=client)
    assert calls[0]["messages"] == [{"role": "user", "content": "question"}]