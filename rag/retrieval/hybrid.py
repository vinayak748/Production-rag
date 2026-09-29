"""
Hybrid retrieval via Reciprocal Rank Fusion (RRF).

RRF combines ranked lists without needing to normalize scores across
retrievers (BM25 scores and cosine similarities aren't on the same scale,
so naive score-averaging is wrong). Each retriever contributes
1 / (k_constant + rank) per item; scores are summed across retrievers.

Reference: Cormack, Clarke, Buettcher (2009) — "Reciprocal Rank Fusion
outperforms Condorcet and individual rank learning methods."

Lite mode: pass use_dense=False to run BM25 only. The dense retriever (and
therefore torch / sentence-transformers) is imported lazily, so it is never
loaded on low-memory deployments.
"""

from .bm25_retriever import BM25Retriever


class HybridRetriever:
    name = "hybrid"

    def __init__(self, rrf_k: int = 60, use_dense: bool = True):
        self.bm25 = BM25Retriever()
        self.dense = None
        if use_dense:
            from .dense_retriever import DenseRetriever  # lazy: heavy import
            self.dense = DenseRetriever()
        self.rrf_k = rrf_k

    def index(self, chunks: dict[str, str]) -> None:
        self.bm25.index(chunks)
        if self.dense is not None:
            self.dense.index(chunks)

    def retrieve(self, query: str, k: int = 10, candidate_pool: int = 50) -> list[tuple[str, float]]:
        bm25_results = self.bm25.retrieve(query, k=candidate_pool)

        if self.dense is None:  # BM25-only (lite mode)
            return bm25_results[:k]

        dense_results = self.dense.retrieve(query, k=candidate_pool)

        rrf_scores: dict[str, float] = {}

        for rank, (chunk_id, _score) in enumerate(bm25_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, (chunk_id, _score) in enumerate(dense_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
