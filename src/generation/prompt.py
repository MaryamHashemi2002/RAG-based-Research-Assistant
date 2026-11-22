NO_ANSWER = "I could not find this in the provided papers."

SYSTEM_PROMPT = f"""You answer questions about research papers using ONLY the numbered sources provided.
Rules:
- Use only information stated in the sources. Do not use outside knowledge.
- If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"
- Cite the sources you used by number in square brackets, like [1] or [2][3].
- Write one short paragraph of 2 to 4 sentences. Do not list the sources and do not copy them word for word."""

MAX_CHARS = 2000  # per source, keeps the whole prompt inside the model's context window


def format_sources(chunks, max_chars= 1000):
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        text = chunk["text"][:max_chars]
        blocks.append(f"[{number}] {chunk['title']} (p.{chunk['page']})\n{text}")
    return "\n\n".join(blocks)


def build_messages(question, chunks, max_chars= 1000):
    user = (
        f"Sources:\n\n{format_sources(chunks, max_chars)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the sources above, and cite them by number."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]