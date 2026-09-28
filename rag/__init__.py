"""
Lazy exports: importing `rag` should not require sentence-transformers/torch
to be installed just to use BM25Retriever, and shouldn't require an
ANTHROPIC_API_KEY just to use retrieval without generation. Each name is
imported only when actually accessed.
"""

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "Reranker",
    "RAGPipeline",
]


def __getattr__(name):
    if name == "BM25Retriever":
        from .retrieval.bm25_retriever import BM25Retriever
        return BM25Retriever
    if name == "DenseRetriever":
        from .retrieval.dense_retriever import DenseRetriever
        return DenseRetriever
    if name == "HybridRetriever":
        from .retrieval.hybrid import HybridRetriever
        return HybridRetriever
    if name == "Reranker":
        from .retrieval.reranker import Reranker
        return Reranker
    if name == "RAGPipeline":
        from .pipeline import RAGPipeline
        return RAGPipeline
    raise AttributeError(f"module 'rag' has no attribute {name!r}")
