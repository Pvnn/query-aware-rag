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

from src.compression.baselines import (
    EXITCompressor,
    RefinerCompressor,
    RECOMPAbstractiveCompressor,
    LLMLingua2Compressor,
    CompactCompressor,
    RecompExtractiveCompressor,
)

from src.generation.reader import RAGReader


class NoOpCompressor:
    def compress(self, query, docs):
        return {"compressed_docs": docs}

# Adapters
class ExitAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class QuitoAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        compressed_docs = []
        for doc in docs:
            sentences = [s.strip() + "." for s in doc.text.split(".") if s.strip()]
            result = self.compressor.compress(query=query, sentences=sentences, compression_ratio=0.5)
            text = " ".join(result["filtered_sentences"])
            compressed_docs.append(SearchResult(evi_id=doc.evi_id, docid=doc.docid, title=doc.title, text=text))
        return {"compressed_docs": compressed_docs}

class HybridAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        compressed_docs = []
        for doc in docs:
            result = self.compressor.compress(query=query, context=doc.text, use_coarse=True, use_fine=True)
            compressed_docs.append(SearchResult(evi_id=doc.evi_id, docid=doc.docid, title=doc.title, text=result['final_text']))
        return {"compressed_docs": compressed_docs}

class RefinerAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class RecompAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class LLMLingua2Adapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class CompActAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class RecompExtractiveAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}


def load_hotpotqa(n=10):
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
        
        df = pd.DataFrame(eval_result["details"])
        csv_path = output_dir / f"details_{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved query details to {csv_path.name}")
        
        return eval_result["aggregate"]

    # --- 1. EXIT Baseline  ---
    exit_model = EXITCompressor(
        token=token,
        base_model="doubleyyh/exit-gemma-2b",  
        cache_dir=None 
    )
    agg = run_and_save("EXIT", ExitAdapter(exit_model))
    results_table.append(format_metrics("EXIT", agg))
    del exit_model
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2. RECOMP Extractive Baseline ---
    # recomp_extr = RecompExtractiveCompressor()
    # agg = run_and_save("RECOMP_EXTR", RecompExtractiveAdapter(recomp_extr))
    # results_table.append(format_metrics("RECOMP_EXTR", agg))
    # del recomp_extr
    # gc.collect()
    # torch.cuda.empty_cache()

    # --- 3. LLMLingua2 Baseline ---
    # llmlingua2 = LLMLingua2Compressor()
    # agg = run_and_save("LLMLingua-2", LLMLingua2Adapter(llmlingua2))
    # results_table.append(format_metrics("LLMLingua-2", agg))
    # del llmlingua2
    # gc.collect()
    # torch.cuda.empty_cache()

    # --- 4. CompAct Baseline ---
    # compact = CompactCompressor(token=token)
    # agg = run_and_save("COMPACT", CompActAdapter(compact))
    # results_table.append(format_metrics("COMPACT", agg))
    # del compact
    # gc.collect()
    # torch.cuda.empty_cache()

    # --- 5. RECOMP Baseline ---
    # recomp = RECOMPAbstractiveCompressor()
    # agg = run_and_save("RECOMP", RecompAdapter(recomp))
    # results_table.append(format_metrics("RECOMP", agg))
    # del recomp
    # gc.collect()
    # torch.cuda.empty_cache()

    # --- 6. REFINER Baseline ---
    # refiner = RefinerCompressor(token=token)
    # agg = run_and_save("REFINER", RefinerAdapter(refiner))
    # results_table.append(format_metrics("REFINER", agg))
    # del refiner
    # gc.collect()
    # torch.cuda.empty_cache()

    # --- 7. NoOp Baseline ---
    # agg = run_and_save("NoOp", NoOpCompressor())
    # results_table.append(format_metrics("NoOp", agg))

    # --- 8. QUITO-X Baseline ---
    # quitox = QuitoxCoarseFilter()
    # agg = run_and_save("QUITO-X", QuitoAdapter(quitox))
    # results_table.append(format_metrics("QUITO-X", agg))
    # del quitox
    # gc.collect()
    # torch.cuda.empty_cache()

    # --- 9. Hybrid Pipeline ---
    # hybrid = HybridCompressor(exit_token=token)
    # agg = run_and_save("HYBRID", HybridAdapter(hybrid))
    # results_table.append(format_metrics("HYBRID", agg))
    # del hybrid
    # gc.collect()
    # torch.cuda.empty_cache()

    # --- Print & Save Final Results ---
    print("\n✅ Generative Benchmark Complete!\n")
    headers = ["Compressor", "EM %", "Token F1 %", "ROUGE-L %", "Disambig-F1 %", "Compression %", "Avg Input Tokens", "Latency (s)"]
    print(tabulate(results_table, headers=headers, tablefmt="github"))
    
    master_df = pd.DataFrame(results_table, columns=headers)
    master_df.to_csv(output_dir / "final_benchmark_results.csv", index=False)
    print("\n✓ Master results saved to eval_results/hotpot_qa/final_benchmark_results.csv")

if __name__ == "__main__":
    run()