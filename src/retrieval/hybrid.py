import sys
from collections import defaultdict

from src.retrieval.bm25 import search_bm25
from src.retrieval.search import close, search

RRF_K = 60   # standard constant from the original RRF paper
POOL = 30    # how many results to take from each method before fusing


def fuse(rankings, k=RRF_K):
    """Merge several ranked lists of chunks with Reciprocal Rank Fusion."""
    scores = defaultdict(float)
    chunks = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            cid = chunk["chunk_id"]
            scores[cid] += 1.0 / (k + rank)
            chunks.setdefault(cid, chunk)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [{**chunks[cid], "score": scores[cid]} for cid in ordered]


def search_hybrid(query, k=5, pool=POOL, **filters):
    dense = search(query, k=pool, **filters)
    sparse = search_bm25(query, k=pool, **filters)
    return fuse([dense, sparse])[:k]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How does L1 regularization select features?"
    for rank, r in enumerate(search_hybrid(question), start=1):
        print(f"{rank}. [{r['score']:.4f}] {r['title']} ({r['year']}, p.{r['page']})")
        print(f"   {r['text'][:200]}...")
    close()