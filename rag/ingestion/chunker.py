"""
Document loading and chunking.

Chunk size/overlap are the first knobs that matter in RAG quality.
Too small -> loses paragraph context, answers feel fragmented.
Too large -> retrieval gets coarse, irrelevant text dilutes the LLM's context.

Defaults below (512 tokens-ish / ~20% overlap) are a reasonable starting
point for prose documents. Re-tune on your own corpus and log what you tried
in the README experiments table.
"""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int


def load_documents(directory: str) -> dict[str, str]:
    """Load all .txt/.md files in a directory into {doc_id: full_text}."""
    docs = {}
    for path in Path(directory).glob("**/*"):
        if path.suffix.lower() in (".txt", ".md"):
            docs[path.stem] = path.read_text(encoding="utf-8", errors="ignore")
    return docs


def chunk_text(
    doc_id: str,
    text: str,
    chunk_size_words: int = 220,   # roughly ~300 tokens for English prose
    overlap_words: int = 40,       # ~20% overlap
) -> list[Chunk]:
    """
    Word-based sliding window chunking. Simple and fast; swap in a
    markdown-header-aware or sentence-boundary-aware splitter if your
    corpus benefits from it (log the before/after in your eval harness
    if you do — that's a stronger resume claim than switching blind).
    """
    words = re.split(r"(\s+)", text)  # keep whitespace so we can reconstruct offsets
    tokens = [w for w in words if w.strip() != ""]

    if not tokens:
        return []

    chunks = []
    step = max(chunk_size_words - overlap_words, 1)
    i = 0
    idx = 0
    while i < len(tokens):
        window = tokens[i : i + chunk_size_words]
        chunk_str = " ".join(window)
        chunks.append(
            Chunk(
                id=f"{doc_id}::chunk_{idx}",
                doc_id=doc_id,
                text=chunk_str,
                start_char=-1,  # not tracked precisely in word-mode; fine for MVP
                end_char=-1,
            )
        )
        idx += 1
        i += step

    return chunks


def chunk_documents(
    docs: dict[str, str],
    chunk_size_words: int = 220,
    overlap_words: int = 40,
) -> dict[str, Chunk]:
    """Chunk every document, return {chunk_id: Chunk}."""
    all_chunks = {}
    for doc_id, text in docs.items():
        for c in chunk_text(doc_id, text, chunk_size_words, overlap_words):
            all_chunks[c.id] = c
    return all_chunks
