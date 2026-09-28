# Hybrid Search in Retrieval-Augmented Generation

Hybrid search combines lexical (keyword-based) retrieval with dense
(embedding-based) semantic retrieval. The two methods have complementary
failure modes: BM25 excels at exact term matching, such as product codes,
names, and rare technical vocabulary, but fails to recognize synonyms or
paraphrased queries. Dense retrieval, based on cosine similarity between
embedding vectors, captures semantic intent and handles paraphrase well,
but frequently underperforms on queries that hinge on a single distinctive
term the embedding model wasn't trained to weight heavily.

Reciprocal Rank Fusion (RRF) is a common technique for combining ranked
lists from multiple retrievers without requiring score normalization.
Each retriever contributes a score of 1 divided by (k + rank) for every
document it returns, where k is a small constant (commonly 60) that
controls how much weight lower-ranked results receive. The contributions
from each retriever are summed, and documents are re-ranked by their total
RRF score. This approach avoids the pitfall of naively averaging BM25 and
cosine similarity scores, which live on incompatible scales.

In production systems, hybrid search consistently outperforms either
method alone, particularly for corpora that mix technical terminology with
natural language questions, such as internal documentation, legal text,
or customer support knowledge bases.
