from src.retrieval.hybrid import fuse


def chunk(cid):
    return {"chunk_id": cid, "score": 0.0}


def test_chunks_in_both_lists_rank_highest():
    dense = [chunk("a"), chunk("b"), chunk("c")]
    sparse = [chunk("c"), chunk("d"), chunk("a")]
    fused = fuse([dense, sparse])
    assert {c["chunk_id"] for c in fused[:2]} == {"a", "c"}


def test_fuse_has_no_duplicates():
    fused = fuse([[chunk("a"), chunk("b")], [chunk("b"), chunk("c")]])
    ids = [c["chunk_id"] for c in fused]
    assert len(ids) == len(set(ids)) == 3


def test_single_ranking_keeps_its_order():
    fused = fuse([[chunk("x"), chunk("y"), chunk("z")]])
    assert [c["chunk_id"] for c in fused] == ["x", "y", "z"]