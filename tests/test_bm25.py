from src.retrieval.bm25 import tokenize

def test_tokenize_lowercases_and_splits_symbols():
    assert tokenize("L2,1-norm Regularization") == ["l2", "1", "norm", "regularization"]

def test_tokenize_drops_stopwords():
    assert tokenize("the lasso and the forest") == ["lasso", "forest"]