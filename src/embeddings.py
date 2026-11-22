from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """
    Converts text into numerical vectors.

    These vectors help the system search by meaning,
    not just exact keywords.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """
        Converts many text chunks into embeddings.
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings.astype("float32")

    def embed_query(self, query: str):
        """
        Converts the user's question into an embedding.
        """

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embedding.astype("float32")