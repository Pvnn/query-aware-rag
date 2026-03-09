import sys
from pathlib import Path
from typing import List, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from tabulate import tabulate

from src.eval.eval_pipeline import CompressorEvaluator
from src.eval.interfaces import SearchResult

from src.compression.hybrid_compressor import HybridCompressor
from src.compression.quitox_filter import QuitoxCoarseFilter
from src.compression.exit_baseline import ExitBaselineCompressor

from dotenv import load_dotenv
import os


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

        result = self.compressor.compress(query, full_text)

        return {
            "compressed_docs": [
                SearchResult(
                    evi_id=0,
                    docid=0,
                    title="exit",
                    text=result["compressed_text"]
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

    # Load environment variables
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    if token is None:
        raise ValueError("HF_TOKEN not found in .env file")

    compressors = {}

    # No compression
    compressors["NoOp"] = NoOpCompressor()

    # EXIT baseline (FIXED: pass token)
    exit_model = ExitBaselineCompressor(token=token)
    compressors["EXIT"] = ExitAdapter(exit_model)

    # QUITO-X
    quitox = QuitoxCoarseFilter()
    compressors["QUITO-X"] = QuitoAdapter(quitox)

    # Hybrid
    hybrid = HybridCompressor(exit_token=token)
    compressors["HYBRID"] = hybrid

    results_table = []

    for name, compressor in compressors.items():

        print(f"\nRunning benchmark for: {name}")

        evaluator = CompressorEvaluator(compressor=compressor)

        metrics = evaluator.evaluate(dataset)

        results_table.append([
            name,
            round(metrics["context_precision"], 2),
            round(metrics["context_recall"], 2),
            round(metrics["context_f1"], 2),
            round(metrics["answer_survival_rate"], 2),
            round(metrics["compression_ratio_chars"], 2),
            round(metrics["avg_latency_sec"], 2),
        ])

    print("\nBenchmark Results:\n")

    headers = [
        "Compressor",
        "Precision %",
        "Recall %",
        "F1 %",
        "Answer Survival %",
        "Compression %",
        "Latency (s)"
    ]

    print(tabulate(results_table, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    run()