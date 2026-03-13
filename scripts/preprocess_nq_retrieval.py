import json
import os
import random
from tqdm import tqdm
import sys
from pathlib import Path

# Ensure we can import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.hybrid_retriever import HybridRetriever

def preprocess_nq():
    input_file = "data/nq/dev.jsonl"
    output_file = "data/nq/nq_top10_hybrid_500.json"
    top_k = 10

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Cannot find {input_file}. Please ensure your previous NQ download script has run.")

    print(f"Loading {input_file}...")
    dataset = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                dataset.append(json.loads(line))
            except Exception:
                pass

    # Randomly sample exactly 500 queries using the standard seed
    print("Sampling 500 queries with seed 42...")
    random.seed(42)
    # Ensure we don't try to sample more items than exist in the dataset
    sample_size = min(500, len(dataset))
    sampled_dataset = random.sample(dataset, sample_size)

    print("Initializing Hybrid Retriever...")
    retriever = HybridRetriever()

    print(f"Filtering {len(sampled_dataset)} queries down to Top {top_k} documents...")
    reduced_dataset = []

    for item in tqdm(sampled_dataset, desc="Processing NQ"):
        query = item.get("question", "")
        
        # 1. Extract the 100 docs
        raw_docs = []
        contexts = item.get("context", item.get("ctxs", []))
        for i, doc in enumerate(contexts):
            title = doc.get("title", f"Doc {i+1}")
            text = doc.get("text", "")
            raw_docs.append({"id": i, "title": title, "text": text})

        # 2. Re-rank with HybridRetriever
        if raw_docs:
            retriever.index_documents(raw_docs)
            retrieved_items = retriever.retrieve(query, top_k=top_k)
            
            top_docs = []
            for doc_dict, score in retrieved_items:
                top_docs.append({
                    "title": doc_dict["title"],
                    "text": doc_dict["text"],
                    "retrieval_score": score
                })
            item["docs"] = top_docs
        else:
            item["docs"] = []

        # Strip the heavy 100-doc array to save file size
        item.pop("ctxs", None)
        item.pop("context", None)

        reduced_dataset.append(item)

    print(f"Saving reduced dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reduced_dataset, f, indent=4)
        
    print("✅ NQ Preprocessing complete!")

if __name__ == "__main__":
    preprocess_nq()