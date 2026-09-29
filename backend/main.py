"""FastAPI wrapper around the existing RAGPipeline.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    POST /reindex        -> (re)loads data/documents and rebuilds the index
    POST /query          -> {"question": "..."} -> pipeline answer
    GET  /health         -> simple liveness check
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Make the project root importable (backend/ sits one level below root)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import RAGPipeline
from rag.ingestion.chunker import load_documents, chunk_documents

app = FastAPI(title="Production RAG API")

# Allow the frontend (served from any origin, e.g. Vercel/Netlify) to call this API.
# Tighten allow_origins to your actual frontend domain before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: RAGPipeline | None = None


def _flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# LITE_MODE=true (default) -> BM25 only, no reranker: fits Render's 512 MB free tier.
# Set LITE_MODE=false on a bigger machine (e.g. Hugging Face Spaces) for the full
# pipeline. USE_DENSE / USE_RERANKER override the LITE_MODE default individually.
LITE_MODE = _flag("LITE_MODE", True)
USE_DENSE = _flag("USE_DENSE", not LITE_MODE)
USE_RERANKER = _flag("USE_RERANKER", not LITE_MODE)


class QueryRequest(BaseModel):
    question: str
    use_reranker: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    source_chunk_ids: list[str]


def build_pipeline() -> RAGPipeline:
    docs = load_documents(str(ROOT / "data" / "documents"))
    chunks = chunk_documents(docs)
    chunk_texts = {cid: c.text for cid, c in chunks.items()}

    p = RAGPipeline(use_reranker=USE_RERANKER, use_dense=USE_DENSE)
    p.index(chunk_texts)
    return p


@app.on_event("startup")
def startup():
    """Build the index once when the server starts, so the first query is fast."""
    global pipeline
    if not os.environ.get("ANTHROPIC_API_KEY"):
        # Don't crash the server if the key is missing -- just fail queries later
        # with a clear error, so /health still works.
        print("WARNING: ANTHROPIC_API_KEY is not set. /query will fail until it is.")
    pipeline = build_pipeline()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed": pipeline is not None,
        "mode": {"bm25": True, "dense": USE_DENSE, "reranker": USE_RERANKER},
    }


@app.post("/reindex")
def reindex():
    global pipeline
    pipeline = build_pipeline()
    return {"status": "reindexed"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not ready yet.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        result = pipeline.query(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        source_chunk_ids=result["source_chunk_ids"],
    )


# Serve the frontend from the same service, so only one Render service is needed.
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
