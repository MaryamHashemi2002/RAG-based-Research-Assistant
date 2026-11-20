from src.retrieval.rerank import rerank_chunks


def chunks():
    return [
        {"chunk_id": "a", "text": "t1"},
        {"chunk_id": "b", "text": "t2"},
        {"chunk_id": "c", "text": "t3"},
    ]


def test_rerank_orders_by_score():
    out = rerank_chunks("q", chunks(), predict=lambda pairs: [0.1, 0.9, 0.5])
    assert [c["chunk_id"] for c in out] == ["b", "c", "a"]


def test_rerank_handles_empty_input():
    assert rerank_chunks("q", [], predict=lambda pairs: []) == []


def test_rerank_pairs_query_with_each_chunk_text():
    seen = []

    def fake(pairs):
        seen.extend(pairs)
        return [0.0] * len(pairs)

    rerank_chunks("my question", chunks(), predict=fake)
    assert seen == [("my question", "t1"), ("my question", "t2"), ("my question", "t3")]