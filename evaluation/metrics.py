"""
Standard IR evaluation metrics, implemented from scratch (no black-box
eval library) so you can actually explain the math in an interview.

All functions take:
    retrieved_ids: list[str]  -- ranked chunk ids returned by your retriever
    relevant_ids: set[str]    -- ground-truth relevant chunk ids for the query
"""

import math


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary relevance NDCG@K."""
    top_k = retrieved_ids[:k]

    dcg = 0.0
    for i, cid in enumerate(top_k, start=1):
        rel = 1 if cid in relevant_ids else 0
        dcg += rel / math.log2(i + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    hits = 0
    precisions = []
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant_ids)


def compute_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int] = [1, 5, 10],
) -> dict:
    return {
        "precision_at_k": {k: precision_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
        "recall_at_k": {k: recall_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
        "ndcg_at_k": {k: ndcg_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
        "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
        "map": average_precision(retrieved_ids, relevant_ids),
    }
