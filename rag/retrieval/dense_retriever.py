"""Dense (embedding-based) retrieval — captures semantic similarity /
paraphrase, where BM25 would miss a query that doesn't share exact words
with the relevant chunk.

Two interchangeable backends, same model (all-MiniLM-L6-v2), so results match:
  - "sentence-transformers" (default): needs torch, ~1 GB RAM
  - "fastembed": ONNX runtime, much lighter; set DENSE_BACKEND=fastembed
"""

import os

import numpy as np


class DenseRetriever:
    name = "dense"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", backend: str | None = None):
        backend = backend or os.environ.get("DENSE_BACKEND", "sentence-transformers")
        self._backend = backend
        if backend == "fastembed":
            from fastembed import TextEmbedding
            self._model = TextEmbedding(f"sentence-transformers/{model_name}")
        else:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        self._chunk_ids: list[str] = []
        self._embeddings: np.ndarray | None = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._backend == "fastembed":
            emb = np.array(list(self._model.embed(texts)))
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / np.clip(norms, 1e-12, None)
        return np.array(
            self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        )

    def index(self, chunks: dict[str, str]) -> None:
        """chunks: {chunk_id: text}"""
        self._chunk_ids = list(chunks.keys())
        texts = [chunks[cid] for cid in self._chunk_ids]
        self._embeddings = self._encode(texts)

    def retrieve(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Returns [(chunk_id, cosine_score), ...] sorted by score desc."""
        if self._embeddings is None:
            raise RuntimeError("Call .index() before .retrieve()")
        q_emb = self._encode([query])[0]
        # embeddings are normalized -> dot product == cosine similarity
        scores = self._embeddings @ q_emb
        top_idx = np.argsort(-scores)[:k]
        return [(self._chunk_ids[i], float(scores[i])) for i in top_idx]
