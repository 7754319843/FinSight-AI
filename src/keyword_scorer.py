import re
from typing import List, Dict


def tokenize(text: str):
    """
    Converts text into simple lowercase tokens.
    """

    return set(
        re.findall(
            r"[a-zA-Z0-9$%.-]+",
            text.lower()
        )
    )


def keyword_score(
    query: str,
    retrieved_chunks: List[Dict],
    top_n: int = 5
) -> List[Dict]:
    """
    Re-scores and re-orders retrieved chunks using keyword overlap.

    How it works:
    1. FAISS finds semantically similar chunks by vector distance.
    2. This function then gives a bonus score to chunks that share
       important words with the question (keyword overlap).
    3. Chunks are re-ordered by this combined score.

    This is a lightweight lexical scoring step — not a neural reranker.
    It improves precision for finance queries where exact terms like
    "revenue", "Q4", or "risk factors" matter a lot.
    """

    query_terms = tokenize(query)

    scored_chunks = []

    for rank, chunk in enumerate(retrieved_chunks):
        chunk_terms = tokenize(chunk["text"])

        keyword_overlap = len(
            query_terms.intersection(chunk_terms)
        )

        # Reciprocal rank from FAISS semantic search
        semantic_score = 1 / (rank + 1)

        final_score = keyword_overlap + semantic_score

        item = dict(chunk)
        item["relevance_score"] = round(final_score, 4)

        scored_chunks.append(item)

    scored_chunks = sorted(
        scored_chunks,
        key=lambda x: x["relevance_score"],
        reverse=True
    )

    return scored_chunks[:top_n]
