# Production RAG: Project Overview

This project implements a hybrid retrieval-augmented generation (RAG)
pipeline designed to go beyond simple vector-similarity search.

## Architecture

The pipeline has four stages: chunking, hybrid retrieval, reranking,
and generation. Documents are split into chunks during ingestion. Each
query is run through two retrievers in parallel: BM25 (a lexical,
keyword-based method) and a dense encoder (a sentence-transformer model
that captures semantic meaning). The two ranked lists are merged using
Reciprocal Rank Fusion (RRF), which combines rankings without needing
to normalize incomparable similarity scores.

## Why hybrid retrieval

Dense embeddings are good at matching paraphrases and intent, but they
can miss exact keyword matches such as product codes, names, or rare
technical terms. BM25 is good at exact matches but fails on paraphrased
queries. Combining both methods covers more query types than either
one alone.

## Reranking

After hybrid retrieval returns a candidate pool (for example, the top
20 chunks), a cross-encoder reranker re-scores each candidate against
the query directly, rather than comparing precomputed embeddings. This
is more accurate but slower, which is why it only runs on the small
candidate pool rather than the entire document collection.

## Evaluation

The project includes a labeled evaluation set of queries mapped to the
chunk IDs that should be retrieved for each one. This is used to
compute Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and
Normalized Discounted Cumulative Gain (NDCG). These metrics make it
possible to measure whether a change, such as adding the reranker,
actually improves retrieval instead of just assuming it does.

## Generation

Generation is decoupled from retrieval in a separate module, so the
underlying LLM provider can be swapped without changing the retrieval
logic. This also allows retrieval quality to be evaluated on its own,
independent of how a specific LLM phrases its final answer.