"""End-to-end RAG pipeline: hybrid retrieval -> rerank -> generate."""

from .retrieval.hybrid import HybridRetriever
from .generation import generate_answer


class RAGPipeline:
    def __init__(self, use_reranker: bool = True, use_dense: bool = True):
        self.retriever = HybridRetriever(use_dense=use_dense)
        self.reranker = None
        if use_reranker:
            from .retrieval.reranker import Reranker  # lazy: heavy import
            self.reranker = Reranker()
        self._chunks: dict[str, str] = {}

    def index(self, chunks: dict[str, str]) -> None:
        """chunks: {chunk_id: text}"""
        self._chunks = chunks
        self.retriever.index(chunks)

    def query(self, question: str, retrieve_k: int = 20, final_k: int = 5) -> dict:
        retrieved = self.retriever.retrieve(question, k=retrieve_k)

        if self.reranker:
            candidates = [(cid, self._chunks[cid]) for cid, _ in retrieved]
            reranked = self.reranker.rerank(question, candidates, top_k=final_k)
            final_ids = [cid for cid, _ in reranked]
        else:
            final_ids = [cid for cid, _ in retrieved[:final_k]]

        context_texts = [self._chunks[cid] for cid in final_ids]
        answer = generate_answer(question, context_texts)

        return {
            "question": question,
            "answer": answer,
            "source_chunk_ids": final_ids,
            "context": context_texts,
        }
