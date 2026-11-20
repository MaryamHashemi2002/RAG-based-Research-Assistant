name,n,hit@1,hit@3,hit@5,hit@10,mrr
dense,16,0.250,0.375,0.438,0.562,0.332
bm25,16,0.125,0.375,0.438,0.750,0.288
hybrid,16,0.250,0.312,0.375,0.688,0.341


- Sentence-aware chunking chosen over fixed (fewer mid-sentence cuts).
- Known noise: author headers, footnotes, a few broken hyphens.
- 25 of 1090 chunks (2.3%) are result tables; the model truncates them at 512 tokens. Acceptable for now; revisit if evaluation (Day 32) shows table questions failing.
- Dense search (bge-small) is on-topic but ranking is imperfect: best L1 answer was rank 5.
- Boilerplate chunks (author lists, copyright text) rank high. Reranker (Day 13) should fix.
- One survey paper dominates results. Consider a per-paper limit later.
- Scores are compressed (0.78-0.84), so no score cutoff for "I don't know".
- Qdrant (local mode) holds 1090 chunks with metadata; results identical to the numpy search.
- Filters by year and paper_id work. Local mode allows one process at a time; use the Qdrant server in Docker on Day 29-30.
- Baseline (dense, 16 questions): hit@1 25%, hit@3 38%, hit@5 44%, hit@10 56%, MRR 0.332.
- In all 10 misses the top result was from the correct paper but a different chunk (mostly p.1 title/abstract). Question wording that names the paper pulls first-page chunks.
- Expectation: reranker (Day 13) helps most.
- split oversized sentences (the 1,404-token chunk bug)
- test 150/250-token chunks
- pass the neighbouring chunks to the LLM (question 16 would have worked)
- try a different reranker
- add a second question set that doesn’t name the paper