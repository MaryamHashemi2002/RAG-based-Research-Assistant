# RAG-based Research Assistant

Ask questions over a collection of research papers and get answers grounded in the source text. The retrieval layer combines dense embeddings and keyword search, and it is evaluated against a hand-labeled question set.

**Status:** retrieval pipeline and evaluation are done. The reranker, LLM answers with citations, API/UI, Docker and CI are in progress.

## What it does today

- Downloads 40 arXiv papers on feature selection and extracts their text
- Cleans the text (removes headers, reference lists, hyphenated line breaks)
- Splits papers into sentence-aware chunks (about 400 tokens, 60 tokens overlap)
- Embeds the chunks with `BAAI/bge-small-en-v1.5` and indexes them in Qdrant
- Searches with dense retrieval, BM25, or a hybrid of both (Reciprocal Rank Fusion)
- Filters results by year or by paper
- Measures retrieval quality with a labeled question set

## Results

Evaluation set: 16 hand-written questions, each with one correct passage. Questions name the paper's topic, so this is a hard test of passage ranking and not of finding the right paper (the right paper is in the top 5 for every question and every method).

| Method | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| Dense (bge-small) | 25% | 38% | 44% | 56% | 0.332 |
| BM25 | 12% | 38% | 44% | 75% | 0.288 |
| Hybrid (RRF) | 25% | 31% | 38% | 69% | 0.341 |

Is the correct passage somewhere in the top N candidates?

| Method | top 10 | top 20 | top 30 | top 50 |
|---|---|---|---|---|
| Dense | 56% | 69% | 69% | 81% |
| BM25 | 75% | 88% | 88% | 94% |
| Hybrid | 62% | 81% | 100% | 100% |

**What this shows**
- Dense search handles loosely worded questions, and BM25 handles exact names and terms. Each method finds questions the other misses.
- Hybrid search doesn't improve the top-5 ranking on this set. Its value is candidate recall: the correct passage is in the top 30 for all 16 questions.
- Most misses are title-and-abstract chunks outranking the passage that answers the question, which is what the next step (a cross-encoder reranker) targets.
- With 16 questions, one question is worth about 6 percentage points. Differences of one or two questions are noise.

## Project structure

```
src/
  ingest/       download_papers, extract_text, clean_text, make_chunks
  retrieval/    embed_chunks, build_index, search, bm25, hybrid
  evaluation/   make_eval_sample, eval_retrieval, pool_recall
eval/           questions.json (labeled questions), results.csv
tests/          unit tests for cleaning, chunking, BM25 and fusion
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Python 3.12 is recommended. On macOS/Linux, activate with `source .venv/bin/activate`.

## Run the pipeline

Run everything from the project root. The `data/` folder is generated and is not stored in the repository.

```
python -m src.ingest.download_papers    # fetch papers from arXiv
python -m src.ingest.extract_text       # PDF -> raw text
python -m src.ingest.clean_text         # clean text, cut reference lists
python -m src.ingest.make_chunks        # sentence-aware chunking
python -m src.retrieval.embed_chunks    # embed chunks (a few minutes on CPU)
python -m src.retrieval.build_index     # build the Qdrant index
```

## Search

```
python -m src.retrieval.search "How does L1 regularization select features?"
python -m src.retrieval.bm25 "How does L1 regularization select features?"
python -m src.retrieval.hybrid "How does L1 regularization select features?"
```

Qdrant runs in local mode (one process at a time). Close other Python shells and debuggers before searching.

## Evaluate and test

```
python -m src.evaluation.eval_retrieval dense    # or: bm25, hybrid
python -m src.evaluation.pool_recall
python -m pytest
```

## Roadmap

- [x] Paper ingestion, text extraction and cleaning
- [x] Sentence-aware chunking
- [x] Dense, BM25 and hybrid retrieval with metadata filters
- [x] Retrieval evaluation (hit@k, MRR, candidate recall) and unit tests
- [ ] Cross-encoder reranker
- [ ] LLM answers with source citations
- [ ] FastAPI backend and Streamlit UI
- [ ] Docker and GitHub Actions CI
- [ ] End-to-end evaluation with RAGAS, tracing with Langfuse

## Known limitations

- About 2% of chunks are result tables or formula-heavy text, which the embedding model truncates at 512 tokens.
- The evaluation set is small and paper-specific. A second set of questions that don't name the paper is planned.
- The hyphen-joining rule can occasionally merge words that should stay hyphenated.

## License

MIT