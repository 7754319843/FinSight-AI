from src.keyword_scorer import keyword_score


class Retriever:
    """
    Combines semantic search (FAISS) with keyword scoring.

    Step 1: FAISS finds the most semantically similar chunks.
    Step 2: Keyword scorer re-orders them by term overlap with the query.
    """

    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = 60,
        rerank_top_n: int = 12
    ):
        """
        Full retrieval process:
        1. Convert question into embedding
        2. Search FAISS for similar chunks
        3. Re-score chunks by keyword overlap
        4. Return best evidence
        """

        query_embedding = self.embedding_model.embed_query(query)

        initial_k = top_k * 2

        initial_results = self.vector_store.search(
            query_embedding,
            top_k=initial_k
        )

        scored_results = keyword_score(
            query=query,
            retrieved_chunks=initial_results,
            top_n=rerank_top_n
        )

        return scored_results