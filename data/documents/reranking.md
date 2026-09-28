# Cross-Encoder Reranking

Bi-encoder models, used in dense retrieval, embed the query and each
document independently into fixed-size vectors, then compare them with a
simple similarity function like cosine distance. This is fast enough to
run over millions of documents but loses information: the model never
sees the query and document together, so it cannot model fine-grained
interactions between specific words in each.

Cross-encoder models solve this by taking the query and a candidate
document as a single joint input and outputting a relevance score
directly. This is far more accurate than bi-encoder similarity, but it is
too slow to run over an entire corpus, since every document requires a
full forward pass through the model for every query.

The standard production pattern is a two-stage pipeline: use a cheap
retriever (BM25, dense, or hybrid) to pull a candidate pool of 20 to 100
documents, then run the cross-encoder only over that small pool to
re-rank it. This captures most of the accuracy benefit of cross-encoders
while keeping latency low enough for real-time queries. A common model
choice for this second stage is a MiniLM-based cross-encoder fine-tuned
on the MS MARCO passage ranking dataset.
