from typing import List, Dict
from pathlib import Path
import pickle

import faiss
import numpy as np


class FaissVectorStore:
    """
    Stores embeddings in FAISS.

    FAISS allows fast similarity search over document chunks.
    """

    def __init__(self):
        self.index = None
        self.metadata = []

    def build(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Builds the FAISS index from chunk embeddings.
        """

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        self.metadata = chunks

    def search(self, query_embedding: np.ndarray, top_k: int = 8) -> List[Dict]:
        """
        Searches for the most similar chunks.
        """

        if self.index is None:
            raise ValueError("Vector store is empty. Build it first.")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for distance, index_position in zip(distances[0], indices[0]):
            if index_position == -1:
                continue

            item = dict(self.metadata[index_position])
            item["distance"] = float(distance)

            results.append(item)

        return results

    def save(self, folder: str):
        """
        Saves FAISS index and metadata.
        """

        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(folder_path / "index.faiss"))

        with open(folder_path / "metadata.pkl", "wb") as file:
            pickle.dump(self.metadata, file)

    def load(self, folder: str):
        """
        Loads FAISS index and metadata.
        """

        folder_path = Path(folder)

        self.index = faiss.read_index(str(folder_path / "index.faiss"))

        with open(folder_path / "metadata.pkl", "rb") as file:
            self.metadata = pickle.load(file)