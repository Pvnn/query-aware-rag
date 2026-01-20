import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict

from src.retrieval.embedder import Embedder


class FaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.texts: List[Dict] = []

    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        self.index.add(embeddings)
        self.texts.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        scores, indices = self.index.search(query_embedding, top_k)
        results = []

        for idx, score in zip(indices[0], scores[0]):
            results.append({
                "score": float(score),
                **self.texts[idx]
            })

        return results


def load_contexts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    contexts = []
    for sample in data:
        for ctx in sample["positive_contexts"] + sample["negative_contexts"]:
            contexts.append({
                "title": ctx["title"],
                "text": ctx["text"]
            })
    return contexts


def build_faiss_index(dev_json_path: str):
    embedder = Embedder()
    contexts = load_contexts(dev_json_path)

    texts = [c["text"] for c in contexts]
    embeddings = embedder.encode(texts)

    index = FaissIndex(dim=embeddings.shape[1])
    index.add(embeddings, contexts)

    return index


if __name__ == "__main__":
   index = build_faiss_index("data/processed/dev.json")
   print(f"FAISS index built with {index.index.ntotal} passages")
