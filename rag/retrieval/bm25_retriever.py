"""BM25 lexical retrieval — catches exact terms, IDs, names that
embeddings tend to blur together."""

from rank_bm25 import BM25Okapi
import re


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    name = "bm25"

    def __init__(self):
        self._bm25 = None
        self._chunk_ids: list[str] = []
        self._chunks: dict[str, str] = {}

    def index(self, chunks: dict[str, str]) -> None:
        """chunks: {chunk_id: text}"""
        self._chunks = chunks
        self._chunk_ids = list(chunks.keys())
        tokenized = [_tokenize(chunks[cid]) for cid in self._chunk_ids]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Returns [(chunk_id, score), ...] sorted by score desc."""
        if self._bm25 is None:
            raise RuntimeError("Call .index() before .retrieve()")
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._chunk_ids, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:k]
