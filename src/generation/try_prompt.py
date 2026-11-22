import sys
import time

from src.generation.llm import MODEL, generate
from src.generation.prompt import build_messages
from src.retrieval.rerank import search_rerank
from src.retrieval.search import close

question = " ".join(sys.argv[1:]) or "How does L1 regularization select features?"

chunks = search_rerank(question, k=5)
messages = build_messages(question, chunks)
print(f"Model: {MODEL}, prompt length: {len(messages[1]['content'])} characters")

start = time.perf_counter()
answer = generate(messages, max_tokens=300)
elapsed = time.perf_counter() - start

print(f"\nANSWER:\n{answer}\n")
print("SOURCES:")
for number, c in enumerate(chunks, start=1):
    print(f"[{number}] {c['title']} (p.{c['page']})")
print(f"\nTook {elapsed:.1f} s")
close()