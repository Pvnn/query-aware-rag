import time
from typing import List, Dict
from tqdm import tqdm

from src.eval.interfaces import SearchResult, BaseCompressor
from src.eval.metrics import context_overlap_scores, has_answer, SimpleTokenizer

class CompressorEvaluator:
    """
    Evaluates Compressor models directly against HotpotQA Gold Contexts.
    This skips the LLM Reader entirely to purely measure compression accuracy.
    """
    
    def __init__(self, compressor: BaseCompressor):
        self.compressor = compressor
        self.tokenizer = SimpleTokenizer()

    def evaluate(self, dataset: List[Dict]) -> Dict:
        """
        Expected HotpotQA Format:
        {
          "question": "str",
          "answer": "str",
          "context": [ ["Title 1", ["Sent 0", "Sent 1"]], ["Title 2", ["Sent 0"]] ],
          "supporting_facts": [ ["Title 1", 0], ["Title 2", 0] ]
        }
        """
        metrics = {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "context_f1": 0.0,
            "answer_survival_rate": 0.0,
            "total_queries": len(dataset)
        }
        
        total_orig_chars = 0
        total_comp_chars = 0
        start_time = time.time()

        for item in tqdm(dataset, desc="Evaluating Compressor"):
            query = item["question"]
            short_answer = item["answer"]
            
            # 1. Build Original Docs & Extract Gold Text
            docs = []
            gold_sentences = []
            
            # Create a map of Title -> List of sentences for fast lookup
            context_map = {title: sentences for title, sentences in item["context"]}
            
            for i, (title, sentences) in enumerate(item["context"]):
                full_doc_text = " ".join(sentences)
                docs.append(SearchResult(evi_id=i, docid=i, title=title, text=full_doc_text))
                
            # Extract exactly what the compressor SHOULD have kept based on HotpotQA labels
            for title, sent_idx in item["supporting_facts"]:
                if title in context_map and sent_idx < len(context_map[title]):
                    gold_sentences.append(context_map[title][sent_idx])
                    
            gold_text = " ".join(gold_sentences)
            orig_text = " ".join([d.text for d in docs])
            total_orig_chars += len(orig_text)

            # 2. Run the Plug-and-Play Compressor
            compressed_docs = self.compressor.compress(query, docs)
            comp_text = " ".join([d.text for d in compressed_docs])
            total_comp_chars += len(comp_text)

            # 3. Calculate Extractive Metrics (Compressor Output vs Gold Text)
            scores = context_overlap_scores(comp_text, gold_text)
            metrics["context_precision"] += scores["precision"]
            metrics["context_recall"] += scores["recall"]
            metrics["context_f1"] += scores["f1"]

            # 4. Answer Survival (Did the raw answer string survive the compression?)
            if has_answer([short_answer], comp_text, self.tokenizer):
                metrics["answer_survival_rate"] += 1

        # --- Final Aggregation ---
        n = metrics["total_queries"]
        if n > 0:
            metrics["context_precision"] = (metrics["context_precision"] / n) * 100
            metrics["context_recall"] = (metrics["context_recall"] / n) * 100
            metrics["context_f1"] = (metrics["context_f1"] / n) * 100
            metrics["answer_survival_rate"] = (metrics["answer_survival_rate"] / n) * 100
            
        metrics["compression_ratio_chars"] = (1 - (total_comp_chars / total_orig_chars)) * 100 if total_orig_chars > 0 else 0.0
        metrics["avg_latency_sec"] = (time.time() - start_time) / n if n > 0 else 0.0

        return metrics