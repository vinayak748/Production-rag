"""
Cross-encoder reranking.

Bi-encoders (used in DenseRetriever) embed query and document separately,
which is fast but loses fine-grained query-document interaction.
Cross-encoders score (query, document) pairs jointly -- much more accurate,
but too slow to run over an entire corpus. The standard pattern: retrieve a
candidate pool cheaply (BM25/dense/hybrid), then rerank only that small pool.
"""

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],  # [(chunk_id, chunk_text), ...]
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [(query, text) for _cid, text in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip([cid for cid, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
