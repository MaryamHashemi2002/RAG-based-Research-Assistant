from src.evaluation.eval_retrieval import is_relevant, load_items
from src.retrieval.hybrid import search_hybrid
from src.retrieval.search import close

items = load_items()
found = sum(
    any(is_relevant(r, item) for r in search_hybrid(item["question"], k=30, pool=30))
    for item in items
)
print(f"Right chunk inside the 30 candidates the reranker sees: {found}/{len(items)}")
close()