# Evaluating Retrieval Quality

Retrieval systems should be evaluated against a labeled test set of
queries paired with known relevant documents, rather than judged purely
by eyeballing a handful of example answers. Several standard metrics
from information retrieval apply directly to RAG systems.

Precision at K measures what fraction of the top K retrieved documents
are actually relevant. Recall at K measures what fraction of all relevant
documents in the corpus were found within the top K results. These two
metrics trade off against each other: retrieving more documents tends to
increase recall while decreasing precision.

Mean Reciprocal Rank (MRR) looks at the position of the first relevant
result in the ranked list; a relevant document at rank 1 scores 1.0,
while one at rank 4 scores 0.25. This is useful when there's typically
only one correct answer and you care about how quickly the system
surfaces it.

Normalized Discounted Cumulative Gain (NDCG) accounts for the position
of every relevant document in the ranking, not just the first one,
weighting documents that appear higher in the list more heavily. It is
normalized against an ideal ranking, producing a score between 0 and 1.

Building an evaluation harness with these metrics, run against a small
hand-labeled set of 20 to 50 queries, is what allows a team to say with
confidence whether a change to chunking, retrieval strategy, or
reranking actually improved the system, rather than assuming it did.
