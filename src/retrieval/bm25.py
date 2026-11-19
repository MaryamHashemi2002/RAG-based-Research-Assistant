import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path("data/chunks_sentence.jsonl")
TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "what", "which", "why", "with", "does", "do", "can",
}

_chunks = None
_bm25 = None


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _load():
    global _chunks, _bm25
    if _bm25 is None:
        with CHUNKS_FILE.open(encoding="utf-8") as f:
            _chunks = [json.loads(line) for line in f]
        _bm25 = BM25Okapi([tokenize(c["text"]) for c in _chunks])
    return _chunks, _bm25


def search_bm25(query, k=5, year_min=None, year_max=None, paper_id=None):
    chunks, bm25 = _load()
    scores = bm25.get_scores(tokenize(query))
    results = []
    for i in np.argsort(scores)[::-1]:
        c = chunks[i]
        if year_min is not None and c["year"] < year_min:
            continue
        if year_max is not None and c["year"] > year_max:
            continue
        if paper_id is not None and c["paper_id"] != paper_id:
            continue
        results.append({"score": float(scores[i]), **c})
        if len(results) == k:
            break
    return results


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How does L1 regularization select features?"
    for rank, r in enumerate(search_bm25(question), start=1):
        print(f"{rank}. [{r['score']:.2f}] {r['title']} ({r['year']}, p.{r['page']})")
        print(f"   {r['text'][:200]}...")