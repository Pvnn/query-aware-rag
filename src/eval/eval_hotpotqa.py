import sys
from pathlib import Path
import gc
import torch
import os
import pandas as pd 
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from tabulate import tabulate

from src.eval.eval_pipeline import GenerativeEvaluator
from src.eval.interfaces import SearchResult

from src.compression.hybrid_compressor import HybridCompressor
from src.compression.quitox_filter import QuitoxCoarseFilter
from src.compression.exit_baseline import ExitBaselineCompressor
from src.generation.reader import RAGReader


class NoOpCompressor:
    def compress(self, query, docs):
        return {"compressed_docs": docs}

class ExitAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
        
    def compress(self, query, docs):
        compressed_docs = []
        for doc in docs:
            # Compress document by document
            result = self.compressor.compress(query, doc.text)
            text = result.get("compressed_text", result.get("final_text", "")) if isinstance(result, dict) else result
            
            compressed_docs.append(
                SearchResult(evi_id=doc.evi_id, docid=doc.docid, title=doc.title, text=text)
            )
        return {"compressed_docs": compressed_docs}

class QuitoAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
        
    def compress(self, query, docs):
        compressed_docs = []
        for doc in docs:
            # Split the individual document into sentences
            sentences = [s.strip() + "." for s in doc.text.split(".") if s.strip()]
            
            result = self.compressor.compress(query=query, sentences=sentences, compression_ratio=0.5)
            text = " ".join(result["filtered_sentences"])
            
            compressed_docs.append(
                SearchResult(evi_id=doc.evi_id, docid=doc.docid, title=doc.title, text=text)
            )
        return {"compressed_docs": compressed_docs}

class HybridAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
        
    def compress(self, query, docs):
        compressed_docs = []
        for doc in docs:
            # Compress document by document, mirroring rag_pipeline.py
            result = self.compressor.compress(
                query=query, 
                context=doc.text, 
                use_coarse=True, 
                use_fine=True,
            )
            
            compressed_docs.append(
                SearchResult(evi_id=doc.evi_id, docid=doc.docid, title=doc.title, text=result['final_text'])
            )
        return {"compressed_docs": compressed_docs}


def load_hotpotqa(n=20):
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    formatted = []
    for i in range(n):
        item = dataset[i]
        context = [[title, sents] for title, sents in zip(item["context"]["title"], item["context"]["sentences"])]
        formatted.append({
            "question": item["question"],
            "answer": item["answer"],
            "context": context
        })
    return formatted


def run():
    print("\nLoading HotpotQA dataset...")
    dataset = load_hotpotqa(n=20) 
    print(f"Loaded {len(dataset)} samples\n")

    output_dir = Path(project_root) / "eval_results" / "hotpot_qa"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Results will be saved to: {output_dir}\n")

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    
    print("Initializing Reader LLM...")
    reader = RAGReader()

    results_table = []

    def format_metrics(name, m):
        return [
            name,
            round(m["em"], 2),
            round(m["token_f1"], 2),
            round(m["rougeL"], 2),
            round(m["disambig_f1"], 2),
            round(m["compression_ratio_chars"], 2),
            round(m.get("avg_prompt_tokens", 0), 1),
            round(m["avg_latency_sec"], 2),
        ]

    def run_and_save(name, adapter):
        print(f"\n[{name}] Running Benchmark...")
        eval_result = GenerativeEvaluator(compressor=adapter, reader=reader).evaluate(dataset)
        
        # Save Details to CSV
        df = pd.DataFrame(eval_result["details"])
        csv_path = output_dir / f"details_{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved query details to {csv_path.name}")
        
        # Return aggregate metrics for the master table
        return eval_result["aggregate"]

    # --- 1. NoOp Baseline ---
    agg = run_and_save("NoOp", NoOpCompressor())
    results_table.append(format_metrics("NoOp", agg))

    # --- 2. EXIT Baseline ---
    exit_model = ExitBaselineCompressor(token=token)
    agg = run_and_save("EXIT", ExitAdapter(exit_model))
    results_table.append(format_metrics("EXIT", agg))
    del exit_model
    gc.collect()
    torch.cuda.empty_cache()

    # --- 3. QUITO-X Baseline ---
    quitox = QuitoxCoarseFilter()
    agg = run_and_save("QUITO-X", QuitoAdapter(quitox))
    results_table.append(format_metrics("QUITO-X", agg))
    del quitox
    gc.collect()
    torch.cuda.empty_cache()

    # --- 4. Hybrid Pipeline ---
    hybrid = HybridCompressor(exit_token=token)
    agg = run_and_save("HYBRID", HybridAdapter(hybrid))
    results_table.append(format_metrics("HYBRID", agg))
    del hybrid
    gc.collect()
    torch.cuda.empty_cache()

    # --- Print & Save Final Results ---
    print("\n✅ Generative Benchmark Complete!\n")
    headers = ["Compressor", "EM %", "Token F1 %", "ROUGE-L %", "Disambig-F1 %", "Compression %", "Avg Input Tokens", "Latency (s)"]
    print(tabulate(results_table, headers=headers, tablefmt="github"))
    
    # Save Master Table
    master_df = pd.DataFrame(results_table, columns=headers)
    master_df.to_csv(output_dir / "final_benchmark_results.csv", index=False)
    print("\n✓ Master results saved to eval_results/hotpot_qa/final_benchmark_results.csv")

if __name__ == "__main__":
    run()