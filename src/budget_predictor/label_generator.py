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
        Soft matching suitable for LLMs.
        This allows:
        - punctuation differences
        - extra words around the answer
        - variations like 'yes', 'Yes', 'Yeah'
        """

        pred = pred.lower().strip()
        gold = gold.lower().strip()

        # remove punctuation
        import re
        pred = re.sub(r"[^a-z0-9\s]", "", pred)
        gold = re.sub(r"[^a-z0-9\s]", "", gold)

        # exact match
        if pred == gold:
            return True

        # gold exists inside pred
        if gold in pred:
            return True

        # simple yes/no normalization
        yes_set = {"yes", "yeah", "yep"}
        no_set = {"no", "nope"}

        if gold in yes_set and pred in yes_set:
            return True
        if gold in no_set and pred in no_set:
            return True

        return False
    
    def find_gold_overlap(self, gold_context: str, retrieved_docs: list[str]):
        """
        Returns list of (doc_index, matching_sentences)
        """
        gold_sents = [s.strip() for s in gold_context.split(".") if s.strip()]
        overlaps = []

        for i, doc in enumerate(retrieved_docs):
            doc_lower = doc.lower()
            matched = []

            for gs in gold_sents:
                if gs.lower() in doc_lower:
                    matched.append(gs)

            if matched:
                overlaps.append((i, matched))

        return overlaps
    

    def find_smallest_k(self, query, gold_answer, gold_context):
        Ks = [2, 4, 6, 8, 10]

        retrieved = self.retriever.retrieve(query, top_k=10)
        docs_text = [d["text"] for d, _ in retrieved]

        for K in Ks:
            selected_docs = docs_text[:K]

            pred = self.reader.generate_answer(query, selected_docs)

            if self.is_correct(pred, gold_answer):
                return K
       
        # Could not generate a correct answer then
        print("\n❌ No correct prediction up to K=10")
        print("Last prediction:", pred)

        overlaps = self.find_gold_overlap(gold_context, docs_text)

        if not overlaps:
            print("\n🚫 Gold context NOT retrieved in top-10")
        else:
            print("\n✅ Gold context WAS retrieved:")
            for doc_id, matched_sents in overlaps:
                print(f"\n[Doc rank {doc_id + 1}]")
                for s in matched_sents:
                    print("  •", s)
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