"""
Benchmark runner: compares BM25-only, dense-only, hybrid, and
hybrid+reranking on your labeled eval set.

This is the script whose output you paste into the README's
"Experiments log" table and turn into resume bullet points.

Usage:
    python examples/build_index.py   # build indexes first
    python evaluation/run_eval.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import BM25Retriever, DenseRetriever, HybridRetriever, Reranker
from rag.ingestion.chunker import load_documents, chunk_documents
from evaluation.metrics import compute_metrics


def load_eval_set(path: str) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def evaluate_retriever(retriever, eval_set: list[dict], k_values: list[int]) -> dict:
    all_metrics = defaultdict(list)

    for example in eval_set:
        query = example["query"]
        relevant_ids = set(example["relevant_chunk_ids"])

        results = retriever.retrieve(query, k=max(k_values))
        retrieved_ids = [cid for cid, _score in results]

        metrics = compute_metrics(retrieved_ids, relevant_ids, k_values)
        for k in k_values:
            all_metrics[f"precision_at_{k}"].append(metrics["precision_at_k"][k])
            all_metrics[f"recall_at_{k}"].append(metrics["recall_at_k"][k])
            all_metrics[f"ndcg_at_{k}"].append(metrics["ndcg_at_k"][k])
        all_metrics["mrr"].append(metrics["mrr"])
        all_metrics["map"].append(metrics["map"])

    return {name: sum(vals) / len(vals) for name, vals in all_metrics.items()}


def evaluate_with_reranking(hybrid, reranker, chunks, eval_set, k_values, rerank_pool=20):
    all_metrics = defaultdict(list)

    for example in eval_set:
        query = example["query"]
        relevant_ids = set(example["relevant_chunk_ids"])

        candidates_raw = hybrid.retrieve(query, k=rerank_pool)
        candidates = [(cid, chunks[cid]) for cid, _ in candidates_raw]
        reranked = reranker.rerank(query, candidates, top_k=max(k_values))
        retrieved_ids = [cid for cid, _score in reranked]

        metrics = compute_metrics(retrieved_ids, relevant_ids, k_values)
        for k in k_values:
            all_metrics[f"precision_at_{k}"].append(metrics["precision_at_k"][k])
            all_metrics[f"recall_at_{k}"].append(metrics["recall_at_k"][k])
            all_metrics[f"ndcg_at_{k}"].append(metrics["ndcg_at_k"][k])
        all_metrics["mrr"].append(metrics["mrr"])
        all_metrics["map"].append(metrics["map"])

    return {name: sum(vals) / len(vals) for name, vals in all_metrics.items()}


def print_table(results_by_strategy: dict[str, dict]):
    strategies = list(results_by_strategy.keys())
    metric_names = list(next(iter(results_by_strategy.values())).keys())

    col_width = 14
    header = "Metric".ljust(20) + "".join(s.ljust(col_width) for s in strategies)
    print(header)
    print("-" * len(header))
    for m in metric_names:
        row = m.ljust(20)
        for s in strategies:
            row += f"{results_by_strategy[s][m]:.3f}".ljust(col_width)
        print(row)


def main():
    k_values = [1, 5, 10]

    docs = load_documents("data/documents")
    if not docs:
        print("No documents found in data/documents/. Add some .txt/.md files first.")
        return

    chunks = chunk_documents(docs)
    chunk_texts = {cid: c.text for cid, c in chunks.items()}

    eval_set_path = "data/eval_set.jsonl"
    if not Path(eval_set_path).exists():
        print(f"No eval set found at {eval_set_path}. See data/eval_set.jsonl for the format.")
        return
    eval_set = load_eval_set(eval_set_path)

    print(f"Loaded {len(chunk_texts)} chunks, {len(eval_set)} eval examples.\n")

    bm25 = BM25Retriever()
    bm25.index(chunk_texts)

    dense = DenseRetriever()
    dense.index(chunk_texts)

    hybrid = HybridRetriever()
    hybrid.bm25 = bm25
    hybrid.dense = dense

    reranker = Reranker()

    results = {
        "bm25": evaluate_retriever(bm25, eval_set, k_values),
        "dense": evaluate_retriever(dense, eval_set, k_values),
        "hybrid": evaluate_retriever(hybrid, eval_set, k_values),
        "hybrid+rerank": evaluate_with_reranking(hybrid, reranker, chunk_texts, eval_set, k_values),
    }

    print_table(results)
    print("\nCopy these numbers into README.md's Experiments log.")


if __name__ == "__main__":
    main()
