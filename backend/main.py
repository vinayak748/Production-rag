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

from fastapi import FastAPI, File, HTTPException, UploadFile
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
GEN_BACKEND = os.environ.get("GEN_BACKEND", "anthropic").strip().lower()


class QueryRequest(BaseModel):
    question: str
    use_reranker: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    source_chunk_ids: list[str]


class UploadResponse(BaseModel):
    status: str
    doc_id: str
    chars_extracted: int
    total_documents: int


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
    key_name = "GEMINI_API_KEY" if GEN_BACKEND == "gemini" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_name):
        # Don't crash the server if the key is missing -- just fail queries later
        # with a clear error, so /health still works.
        print(f"WARNING: {key_name} is not set. /query will fail until it is.")
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


def _extract_text(filename: str, raw: bytes) -> str:
    """Extract plain text from an uploaded file. txt/md read as-is; pdf via pypdf."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".txt", ".md"):
        return raw.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="pypdf is not installed on the server; add it to requirements to enable PDF uploads.",
            )
        import io

        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type '{suffix}'. Upload .txt, .md, or .pdf.",
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Add a new document to the corpus and rebuild the index.

    Saves the extracted text under data/documents/<name>.md and reindexes,
    so the document is searchable immediately -- no server restart needed.
    """
    raw = await file.read()
    text = _extract_text(file.filename, raw)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in the uploaded file.")

    doc_id = Path(file.filename).stem
    docs_dir = ROOT / "data" / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / f"{doc_id}.md").write_text(text, encoding="utf-8")

    global pipeline
    pipeline = build_pipeline()

    return UploadResponse(
        status="indexed",
        doc_id=doc_id,
        chars_extracted=len(text),
        total_documents=len(load_documents(str(docs_dir))),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not ready yet.")
    key_name = "GEMINI_API_KEY" if GEN_BACKEND == "gemini" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_name):
        raise HTTPException(status_code=500, detail=f"{key_name} is not set on the server.")
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
