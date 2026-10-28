from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from energy_intelligence.rag import hybrid_search


@pytest.fixture
def corpus():
    return [
        (1, "RTE.pdf", 1, "nuclear generation rose"),
        (2, "RTE.pdf", 2, "solar generation increased"),
        (3, "RTE.pdf", 3, "wind output declined"),
    ]


def connection(corpus):
    conn = MagicMock()
    conn.execute.side_effect = lambda sql, *args: MagicMock(fetchall=lambda: (
        list(reversed(corpus)) if "<=>" in sql else list(corpus)
    ))
    return conn


def model():
    embedding = MagicMock()
    embedding.get_query_embedding.return_value = [1.0, 0.0]
    return embedding


def test_repeated_queries_reuse_corpus_and_bm25(corpus, monkeypatch):
    conn = connection(corpus)
    build_bm25 = MagicMock(wraps=hybrid_search.BM25Okapi)
    monkeypatch.setattr(hybrid_search, "BM25Okapi", build_bm25)
    retriever = hybrid_search.HybridRetriever()
    first = retriever.retrieve("nuclear", conn, model())
    second = retriever.retrieve("nuclear", conn, model())
    assert first == second
    assert first[0]["id"] == 1
    assert build_bm25.call_count == 1
    assert sum("<=>" not in call.args[0] for call in conn.execute.call_args_list) == 1
    assert sum("<=>" in call.args[0] for call in conn.execute.call_args_list) == 2


def test_new_instance_reloads_new_documents(corpus):
    conn = connection(corpus)
    first = hybrid_search.HybridRetriever()
    first.retrieve("nuclear", conn, model())
    corpus.append((4, "new.pdf", 1, "hydro output"))
    second = hybrid_search.HybridRetriever()
    result = second.retrieve("hydro", conn, model())
    assert any(row["id"] == 4 for row in result)
    assert len(second._rows) == 4
    assert len(first._rows) == 3


def test_new_vector_match_does_not_fail_with_cached_corpus(corpus):
    conn = connection(corpus)
    retriever = hybrid_search.HybridRetriever()
    retriever.retrieve("nuclear", conn, model())
    corpus.append((4, "new.pdf", 7, "hydro output"))
    results = retriever.retrieve("hydro", conn, model())
    new_chunk = next(row for row in results if row["id"] == 4)
    assert new_chunk["source"] == "new.pdf"
    assert new_chunk["page"] == 7


def test_concurrent_first_queries_load_once(corpus, monkeypatch):
    conn = connection(corpus)
    build_bm25 = MagicMock(wraps=hybrid_search.BM25Okapi)
    monkeypatch.setattr(hybrid_search, "BM25Okapi", build_bm25)
    retriever = hybrid_search.HybridRetriever()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: retriever.retrieve("solar", conn, model()), range(4)))
    assert all(result == results[0] for result in results)
    assert build_bm25.call_count == 1


def test_empty_corpus_does_not_call_embedding_model():
    conn = connection([])
    embedding = model()
    assert hybrid_search.HybridRetriever().retrieve("nuclear", conn, embedding) == []
    embedding.get_query_embedding.assert_not_called()


def test_tokenless_corpus_still_returns_vector_matches():
    conn = connection([(1, "RTE.pdf", 1, "the and of")])
    result = hybrid_search.HybridRetriever().retrieve("nuclear", conn, model())
    assert result[0]["id"] == 1


def test_standalone_wrapper_does_not_share_cache(corpus):
    conn = connection(corpus)
    hybrid_search.hybrid_retrieve("nuclear", conn, model())
    hybrid_search.hybrid_retrieve("solar", conn, model())
    assert sum("<=>" not in call.args[0] for call in conn.execute.call_args_list) == 2
