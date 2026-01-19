import json
from datasets import load_from_disk
from pathlib import Path

INPUT_PATH = "data/hotpotqa/dev"
OUTPUT_PATH = "data/processed"
OUTPUT_FILE = "dev.json"


def main():
    dataset = load_from_disk(INPUT_PATH)
    processed = []

    for sample in dataset:
        question = sample["question"]
        answer = sample["answer"]

        positive_contexts = []
        negative_contexts = []

        supporting_titles = set(sample["supporting_facts"]["title"])

        for title, sentences in zip(
            sample["context"]["title"],
            sample["context"]["sentences"]
        ):
            text = " ".join(sentences)

            if title in supporting_titles:
                positive_contexts.append({
                    "title": title,
                    "text": text
                })
            else:
                negative_contexts.append({
                    "title": title,
                    "text": text
                })

        if len(positive_contexts) >= 2:
            processed.append({
                "question": question,
                "answer": answer,
                "positive_contexts": positive_contexts,
                "negative_contexts": negative_contexts
            })

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    with open(f"{OUTPUT_PATH}/{OUTPUT_FILE}", "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)

    print(f"Saved {len(processed)} samples to {OUTPUT_PATH}/{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
