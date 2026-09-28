"""Loads documents from data/documents/, chunks them, and reports stats.
The RAGPipeline builds its own indexes on .index() -- this script is just
for sanity-checking your chunking before you run a full query or eval."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.ingestion.chunker import load_documents, chunk_documents


def main():
    docs = load_documents("data/documents")
    print(f"Loaded {len(docs)} documents: {list(docs.keys())}")

    chunks = chunk_documents(docs)
    print(f"Produced {len(chunks)} chunks.\n")

    for cid, chunk in list(chunks.items())[:3]:
        print(f"--- {cid} ---")
        print(chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""))
        print()


if __name__ == "__main__":
    main()
