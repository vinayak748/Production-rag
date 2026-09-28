# Production RAG — Hybrid Search + Reranking + Evaluation

A retrieval-augmented generation pipeline built to production-grade standards:
hybrid retrieval (BM25 + dense embeddings, fused with Reciprocal Rank Fusion),
cross-encoder reranking, and a proper evaluation harness — not just a
"call an LLM API" demo.

## Why this exists

Most RAG tutorials stop at "embed chunks, cosine-similarity search, stuff into
a prompt." That breaks the moment a user searches for an exact term, a product
code, or a name the embedding model doesn't represent well. This project fixes
that with hybrid retrieval, and — more importantly — it *measures* whether
each addition actually helps, instead of assuming it does.

## Architecture

```
Documents (.txt/.md/.pdf)
    │
    ▼
Chunking (fixed-size + overlap, configurable)
    │
    ▼
┌─────────────┬─────────────┐
│ BM25 Index  │ Dense Index │  (built in parallel)
└──────┬──────┴──────┬──────┘
       │             │
       ▼             ▼
   Keyword       Semantic
   Retrieval     Retrieval
       │             │
       └──────┬──────┘
              ▼
   Reciprocal Rank Fusion (hybrid)
              │
              ▼
     Cross-Encoder Reranker
              │
              ▼
        Top-K Context
              │
              ▼
          LLM Answer
              │
              ▼
     Evaluation Harness
  (retrieval + answer quality)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your LLM API key (used only for the generation step, not retrieval):

```bash
export ANTHROPIC_API_KEY=your_key_here
```

## Quick start

```bash
# 1. Ingest sample documents and build indexes
python examples/build_index.py

# 2. Ask a question end-to-end
python examples/query.py "What is reciprocal rank fusion?"

# 3. Run the evaluation harness (retrieval metrics, no LLM calls needed)
python examples/run_eval.py
```

## What makes this "production-grade" and not a toy

1. **Hybrid retrieval, not just vector search.** BM25 catches exact terms,
   IDs, and rare words that embeddings blur together; dense retrieval catches
   paraphrases and intent. Fused with RRF so neither dominates.
2. **Reranking is measured, not assumed to help.** `evaluation/run_eval.py`
   reports retrieval metrics with and without the reranker, so you can see
   the actual delta on your own data — this is the number to put on your
   resume, not a generic claim.
3. **A real eval set.** `data/eval_set.jsonl` holds hand-labeled
   query → relevant-chunk-ids pairs. Fill this in with ~30 examples from your
   own corpus before you claim results.
4. **Explicit chunking tradeoffs.** `rag/ingestion/chunker.py` supports
   configurable chunk size / overlap — document what you tried and why you
   landed on your final values (see "Experiments" below).
5. **Answer faithfulness, not just retrieval quality.** `evaluation/faithfulness.py`
   checks whether the generated answer is actually grounded in the retrieved
   context, using a lightweight LLM-as-judge prompt — catches the case where
   the model answers correctly from its own knowledge while ignoring what you
   retrieved.

## Project structure

```
production-rag/
├── rag/
│   ├── ingestion/
│   │   └── chunker.py          # document loading + chunking
│   ├── retrieval/
│   │   ├── bm25_retriever.py   # lexical search
│   │   ├── dense_retriever.py  # embedding-based search
│   │   ├── hybrid.py           # RRF fusion
│   │   └── reranker.py         # cross-encoder reranking
│   ├── generation.py           # LLM call with retrieved context
│   └── pipeline.py             # ties it all together
├── evaluation/
│   ├── metrics.py              # precision/recall/MRR/NDCG
│   ├── run_eval.py             # retrieval benchmark runner
│   └── faithfulness.py         # LLM-as-judge answer grounding check
├── examples/
│   ├── build_index.py
│   └── query.py
├── data/
│   ├── documents/               # put your source docs here
│   └── eval_set.jsonl           # hand-labeled query/relevant-doc pairs
└── requirements.txt
```

## Experiments log (fill this in as you build)

Keep a running log here — this is what you'll actually talk about in
interviews. Example format:

| Change | Metric | Before | After |
|---|---|---|---|
| Added BM25 (vs. dense-only) | Recall@10 | 0.61 | 0.78 |
| Added reranking | Precision@5 | 0.55 | 0.71 |
| Chunk size 256 → 512 | NDCG@10 | 0.64 | 0.70 |

## Resume bullet points (edit with your real numbers once you've run evals)

- Built a hybrid retrieval pipeline (BM25 + dense embeddings fused via
  Reciprocal Rank Fusion) with cross-encoder reranking, improving
  Recall@10 by X% and Precision@5 by Y% over baseline vector search
- Designed and ran a retrieval evaluation harness (Precision/Recall/MRR/NDCG@K)
  against a hand-labeled test set to quantify the impact of each pipeline change
- Implemented an LLM-as-judge faithfulness check to catch answers not grounded
  in retrieved context, reducing ungrounded responses by Z%
