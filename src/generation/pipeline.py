import sys
import time
from dataclasses import dataclass, field

from src.generation.llm import generate
from src.generation.prompt import NO_ANSWER, build_messages
from src.retrieval.rerank import search_rerank
from src.retrieval.search import close

TOP_K = 3
EMPTY_MESSAGE = "The model returned an empty answer. Please try again."


@dataclass
class Answer:
    question: str
    text: str
    status: str  # ok, no_answer, no_results, empty_answer, empty_question, llm_error
    sources: list = field(default_factory=list)


def clean_answer(raw):
    """Turn the model's raw output into (text, status)."""
    text = (raw or "").strip()
    if not text:
        return EMPTY_MESSAGE, "empty_answer"
    if NO_ANSWER.lower() in text.lower():
        return NO_ANSWER, "no_answer"  # drop any extra text the model added around the refusal
    return text, "ok"


def _default_llm(messages):
    return generate(messages, temperature=0, max_tokens=300)


def ask(question, k=TOP_K, retrieve=None, llm=None, **filters):
    question = (question or "").strip()
    if not question:
        return Answer(question, "Please enter a question.", "empty_question")

    retrieve = retrieve or search_rerank
    llm = llm or _default_llm

    chunks = retrieve(question, k=k, **filters)
    if not chunks:
        return Answer(question, NO_ANSWER, "no_results")

    try:
        raw = llm(build_messages(question, chunks))
    except Exception as error:
        message = f"The language model is not available ({type(error).__name__}). Is Ollama running?"
        return Answer(question, message, "llm_error")

    text, status = clean_answer(raw)
    sources = []
    if status == "ok":
        sources = [
            {
                "n": number,
                "title": c["title"],
                "page": c["page"],
                "year": c.get("year"),
                "paper_id": c.get("paper_id"),
            }
            for number, c in enumerate(chunks, start=1)
        ]
    return Answer(question, text, status, sources)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How does L1 regularization select features?"
    start = time.perf_counter()
    result = ask(question)
    print(f"[{result.status}] {result.text}")
    for s in result.sources:
        print(f"  [{s['n']}] {s['title']} (p.{s['page']})")
    print(f"\nTook {time.perf_counter() - start:.1f} s")
    close()