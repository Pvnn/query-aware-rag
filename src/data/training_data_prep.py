import sys
from pathlib import Path

# Add project root to PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_from_disk
import json
from tqdm import tqdm


class HotpotQATrainingDataGenerator:
    def __init__(self, data_path="data/hotpotqa/train"):
        print(f"Loading HotpotQA from {data_path}...")
        self.dataset = load_from_disk(data_path)
        print(f"✓ Loaded {len(self.dataset)} training examples")

    def create_training_samples(self, max_samples=10000):
        """
        Returns list of dicts with:
        query, sentence, document, label
        """
        samples = []

        print(f"Generating training samples (max {max_samples})...")

        for idx in tqdm(range(min(len(self.dataset), max_samples))):
            item = self.dataset[idx]

            query = item["question"]

            # ---- FIX: normalize supporting facts ----
            sf_titles = item["supporting_facts"]["title"]
            sf_sent_ids = item["supporting_facts"]["sent_id"]
            supporting_pairs = list(zip(sf_titles, sf_sent_ids))

            context_titles = item["context"]["title"]
            context_sentences = item["context"]["sentences"]

            # Build document map
            doc_map = {
                title: sents
                for title, sents in zip(context_titles, context_sentences)
            }

            # --------------------
            # Positive samples
            # --------------------
            for title, sent_id in supporting_pairs:
                if title in doc_map and sent_id < len(doc_map[title]):
                    samples.append({
                        "query": query,
                        "sentence": doc_map[title][sent_id],
                        "document": " ".join(doc_map[title]),
                        "label": "Yes"
                    })

            # --------------------
            # Hard negatives
            # --------------------
            for title, sent_id in supporting_pairs:
                if title in doc_map:
                    for i, sent in enumerate(doc_map[title]):
                        if i != sent_id:
                            samples.append({
                                "query": query,
                                "sentence": sent,
                                "document": " ".join(doc_map[title]),
                                "label": "No"
                            })
                            break

            # --------------------
            # Random negatives
            # --------------------
            supporting_titles = set(sf_titles)

            for title in context_titles:
                if title not in supporting_titles and len(doc_map[title]) > 0:
                    samples.append({
                        "query": query,
                        "sentence": doc_map[title][0],
                        "document": " ".join(doc_map[title]),
                        "label": "No"
                    })
                    break

        print(f"\n✓ Generated {len(samples)} training samples")

        pos = sum(1 for s in samples if s["label"] == "Yes")
        neg = len(samples) - pos

        print(f"  Positive: {pos}")
        print(f"  Negative: {neg}")
        print(f"  Ratio: {pos / len(samples):.2%} positive")

        return samples

    def save_samples(self, samples, output_path="data/training_samples.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
        print(f"✓ Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    generator = HotpotQATrainingDataGenerator()
    samples = generator.create_training_samples(max_samples=1000)
    generator.save_samples(samples)
