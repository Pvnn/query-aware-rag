import numpy as np
from tqdm import tqdm

from src.budget_predictor.metadata_extractor import MetadataExtractor


class LabelGenerator:
    """
    Generates true K labels using a retriever + Ollama reader.
    """

    def __init__(self, retriever, reader):
        self.retriever = retriever
        self.reader = reader
        self.extractor = MetadataExtractor()

    def is_correct(self, pred, gold):
        """
        Basic exact-match style normalization.
        """
        pred = pred.strip().lower()
        gold = gold.strip().lower()
        return pred == gold

    def find_smallest_k(self, query, gold_answer):
        Ks = [2, 4, 6, 8]

        retrieved = self.retriever.retrieve(query, top_k=10)
        docs_text = [d["text"] for d, _ in retrieved]

        for K in Ks:
            selected_docs = docs_text[:K]

            pred = self.reader.answer(query, selected_docs)

            if self.is_correct(pred, gold_answer):
                return K

        return None

    def generate_training_pairs(self, dataset, max_samples=500):
        """
        Returns list of:
        {
            "features": [46 floats],
            "label": K
        }
        """
        output = []

        for item in tqdm(dataset[:max_samples], desc="Labeling"):
            query = item["question"]
            answer = item["answer"]

            retrieved = self.retriever.retrieve(query, top_k=10)
            docs_text = [d["text"] for d, _ in retrieved]
            docs_scores = [score for _, score in retrieved]

            # build features
            features = self.extractor.extract(
                query,
                [{"text": t, "score": s} for t, s in zip(docs_text, docs_scores)]
            )

            K = self.find_smallest_k(query, answer)
            if K is None:
                continue

            output.append({
                "features": features.tolist(),
                "label": K
            })

        return output