"""End-to-end query: index documents, run hybrid retrieval + reranking,
generate an answer. Requires ANTHROPIC_API_KEY to be set for the
generation step (retrieval/reranking work without it)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import RAGPipeline
from rag.ingestion.chunker import load_documents, chunk_documents


def main():
    if len(sys.argv) < 2:
        print('Usage: python examples/query.py "your question here"')
        return

    question = sys.argv[1]

    docs = load_documents("data/documents")
    chunks = chunk_documents(docs)
    chunk_texts = {cid: c.text for cid, c in chunks.items()}

    print(f"Indexing {len(chunk_texts)} chunks...")
    pipeline = RAGPipeline(use_reranker=True)
    pipeline.index(chunk_texts)

    print(f"\nQuery: {question}\n")
    result = pipeline.query(question)

    print("Retrieved chunks:", result["source_chunk_ids"])
    print("\nAnswer:\n", result["answer"])


if __name__ == "__main__":
    main()
