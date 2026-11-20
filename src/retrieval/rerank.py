import sys
import time

from sentence_transformers import CrossEncoder

from src.retrieval.hybrid import search_hybrid
from src.retrieval.search import close

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
POOL = 30  # candidates taken from hybrid search before reranking

_model = None


def _predict(pairs):
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME, max_length=512)
    return _model.predict(pairs, batch_size=16, show_progress_bar=False)


def rerank_chunks(query, chunks, predict=None):
    """Re-sort chunks by how well each one answers the query."""
    if not chunks:
        return []
    predict = predict or _predict
    scores = predict([(query, c["text"]) for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return [{**chunk, "score": float(score)} for score, chunk in ranked]


def search_rerank(query, k=5, pool=POOL, **filters):
    candidates = search_hybrid(query, k=pool, pool=pool, **filters)
    return rerank_chunks(query, candidates)[:k]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How does L1 regularization select features?"
    search_rerank(question)  # warm-up call: loads the models
    start = time.perf_counter()
    results = search_rerank(question)
    elapsed = time.perf_counter() - start
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. [{r['score']:.2f}] {r['title']} ({r['year']}, p.{r['page']})")
        print(f"   {r['text'][:200]}...")
    print(f"\nSearch + rerank took {elapsed:.2f} s")
    close()