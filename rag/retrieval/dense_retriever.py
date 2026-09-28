"""Dense (embedding-based) retrieval — captures semantic similarity /
paraphrase, where BM25 would miss a query that doesn't share exact words
with the relevant chunk."""

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    name = "dense"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._chunk_ids: list[str] = []
        self._embeddings: np.ndarray | None = None

    def index(self, chunks: dict[str, str]) -> None:
        """chunks: {chunk_id: text}"""
        self._chunk_ids = list(chunks.keys())
        texts = [chunks[cid] for cid in self._chunk_ids]
        embeddings = self._model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        self._embeddings = np.array(embeddings)

    def retrieve(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Returns [(chunk_id, cosine_score), ...] sorted by score desc."""
        if self._embeddings is None:
            raise RuntimeError("Call .index() before .retrieve()")
        q_emb = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )[0]
        # embeddings are normalized -> dot product == cosine similarity
        scores = self._embeddings @ q_emb
        top_idx = np.argsort(-scores)[:k]
        return [(self._chunk_ids[i], float(scores[i])) for i in top_idx]
