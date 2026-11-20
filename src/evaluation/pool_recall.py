from src.evaluation.eval_retrieval import is_relevant, load_items
from src.retrieval.bm25 import search_bm25
from src.retrieval.hybrid import fuse
from src.retrieval.search import close, search

POOLS = [10, 20, 30, 50]


def main():
    items = load_items()
    found = {name: {p: 0 for p in POOLS} for name in ("dense", "bm25", "hybrid")}

    for item in items:
        dense = search(item["question"], k=max(POOLS))
        sparse = search_bm25(item["question"], k=max(POOLS))
        fused = fuse([dense, sparse])
        for p in POOLS:
            for name, ranking in (("dense", dense), ("bm25", sparse), ("hybrid", fused)):
                if any(is_relevant(r, item) for r in ranking[:p]):
                    found[name][p] += 1

    n = len(items)
    print(f"\nRight chunk found within the top N candidates ({n} questions)")
    print("method   " + "  ".join(f"top{p:<3}" for p in POOLS))
    for name in found:
        print(f"{name:<8} " + "  ".join(f"{found[name][p] / n:>6.0%}" for p in POOLS))
    close()


if __name__ == "__main__":
    main()