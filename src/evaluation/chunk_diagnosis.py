import json

from src.evaluation.eval_retrieval import is_relevant, load_items
from src.retrieval.hybrid import search_hybrid
from src.retrieval.search import close

chunks = {}
with open("data/chunks_sentence.jsonl", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        chunks[c["chunk_id"]] = c

print("id  hybrid_rank  chunk_tokens  answer_position_in_chunk")
in_pool = 0
items = load_items()
for item in items:
    c = chunks[item["chunk_id"]]
    pos = c["text"].lower().find(item["keyphrase"].strip().lower()) / len(c["text"])
    results = search_hybrid(item["question"], k=30, pool=30)
    rank = next((r for r, res in enumerate(results, 1) if is_relevant(res, item)), None)
    in_pool += rank is not None
    print(f"{item['id']:>2}  {str(rank):>11}  {c['n_tokens']:>12}  {pos:>10.0%}")

print(f"\nRight chunk inside the 30 candidates: {in_pool}/{len(items)}")
close()