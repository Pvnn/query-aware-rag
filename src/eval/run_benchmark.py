import sys
from pathlib import Path
import gc
import torch
import os
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from tabulate import tabulate

from src.eval.eval_pipeline import CompressorEvaluator
from src.eval.interfaces import SearchResult

from src.compression.hybrid_compressor import HybridCompressor
from src.compression.quitox_filter import QuitoxCoarseFilter
from src.compression.exit_baseline import ExitBaselineCompressor

# -----------------------------
# Baseline: No Compression
# -----------------------------
class NoOpCompressor:
    def compress(self, query, docs):
        return {"compressed_docs": docs}

# -----------------------------
# Adapter for EXIT baseline
# -----------------------------
class ExitAdapter:
    def __init__(self, compressor):
        self.compressor = compressor

    def compress(self, query, docs):
        full_text = " ".join([d.text for d in docs])
        
        # Call compressor
        result = self.compressor.compress(query, full_text)
        
        # FIX: Handle if the model returns a raw string OR a dictionary
        if isinstance(result, dict):
            text = result.get("compressed_text", result.get("final_text", ""))
        else:
            text = result  # If it's just the raw string

        return {
            "compressed_docs": [
                SearchResult(
                    evi_id=0,
                    docid=0,
                    title="exit",
                    text=text
                )
            ]
        }

# -----------------------------
# Adapter for QUITO-X
# -----------------------------
class QuitoAdapter:
    def __init__(self, compressor):
        self.compressor = compressor

    def compress(self, query, docs):
        sentences = []
        for d in docs:
            sentences.extend(d.text.split("."))

        result = self.compressor.compress(
            query=query,
            sentences=sentences,
            compression_ratio=0.7
        )

        text = " ".join(result["filtered_sentences"])

        return {
            "compressed_docs": [
                SearchResult(
                    evi_id=0,
                    docid=0,
                    title="quito",
                    text=text
                )
            ]
        }

# -----------------------------
# Adapter for HYBRID
# -----------------------------
class HybridAdapter:
    def __init__(self, compressor):
        self.compressor = compressor

    def compress(self, query, docs):
        full_text = " ".join([d.text for d in docs])
        
        # Hybrid compressor expects standard string inputs and strict kwargs
        result = self.compressor.compress(
            query=query, 
            context=full_text,
            coarse_ratio=0.7,
            fine_threshold=0.5,
            use_coarse=True,
            use_fine=True
        )
        
        # Hybrid returns a detailed dict containing 'final_text'
        return {
            "compressed_docs": [
                SearchResult(
                    evi_id=0,
                    docid=0,
                    title="hybrid",
                    text=result['final_text']
                )
            ]
        }

# -----------------------------
# Load HotpotQA dataset
# -----------------------------
def load_hotpotqa(n=100):
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    formatted = []
    for i in range(n):
        item = dataset[i]
        context = []
        for title, sents in zip(item["context"]["title"], item["context"]["sentences"]):
            context.append([title, sents])
        supporting = []
        for t, idx in zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"]):
            supporting.append([t, idx])
        formatted.append({
            "question": item["question"],
            "answer": item["answer"],
            "context": context,
            "supporting_facts": supporting
        })
    return formatted

# -----------------------------
# Benchmark Runner
# -----------------------------
def run():
    print("\nLoading HotpotQA dataset...")
    dataset = load_hotpotqa(n=100)
    print(f"Loaded {len(dataset)} samples\n")

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN not found in .env file")

    results_table = []

    def format_metrics(name, m):
        return [
            name,
            round(m["context_precision"], 2),
            round(m["context_recall"], 2),
            round(m["context_f1"], 2),
            round(m["answer_survival_rate"], 2),
            round(m["compression_ratio_chars"], 2),
            round(m["avg_latency_sec"], 2),
        ]

    # --- 1. NoOp Baseline ---
    print("\n=====================================")
    print(" Running Benchmark: NoOp (Baseline)")
    print("=====================================")
    noop = NoOpCompressor()
    metrics = CompressorEvaluator(compressor=noop).evaluate(dataset)
    results_table.append(format_metrics("NoOp", metrics))


    # --- 2. EXIT Baseline ---
    print("\n=====================================")
    print(" Running Benchmark: EXIT")
    print("=====================================")
    exit_model = ExitBaselineCompressor(token=token)
    metrics = CompressorEvaluator(compressor=ExitAdapter(exit_model)).evaluate(dataset)
    results_table.append(format_metrics("EXIT", metrics))
    
    # 🧹 FREE GPU MEMORY
    del exit_model
    gc.collect()
    torch.cuda.empty_cache()


    # --- 3. QUITO-X Baseline ---
    print("\n=====================================")
    print(" Running Benchmark: QUITO-X")
    print("=====================================")
    quitox = QuitoxCoarseFilter()
    metrics = CompressorEvaluator(compressor=QuitoAdapter(quitox)).evaluate(dataset)
    results_table.append(format_metrics("QUITO-X", metrics))
    
    # 🧹 FREE GPU MEMORY
    del quitox
    gc.collect()
    torch.cuda.empty_cache()


    # --- 4. Hybrid Pipeline ---
    print("\n=====================================")
    print(" Running Benchmark: HYBRID")
    print("=====================================")
    hybrid = HybridCompressor(exit_token=token)
    metrics = CompressorEvaluator(compressor=HybridAdapter(hybrid)).evaluate(dataset)
    results_table.append(format_metrics("HYBRID", metrics))

    # 🧹 FREE GPU MEMORY
    del hybrid
    gc.collect()
    torch.cuda.empty_cache()


    # --- Print Final Results ---
    print("\n✅ Benchmark Complete!\n")
    headers = [
        "Compressor", "Precision %", "Recall %", "F1 %", 
        "Answer Survival %", "Compression %", "Latency (s)"
    ]
    print(tabulate(results_table, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    run()