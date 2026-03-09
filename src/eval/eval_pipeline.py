import time
from typing import List, Dict
from tqdm import tqdm

from src.eval.interfaces import SearchResult
from src.eval.metrics import context_overlap_scores, has_answer, SimpleTokenizer


class CompressorEvaluator:
    """
    Evaluates compressors directly against HotpotQA gold contexts.
    No adapters required — compressors must return {"compressed_docs": [...]}
    """

    def __init__(self, compressor):
        self.compressor = compressor
        self.tokenizer = SimpleTokenizer()

    def evaluate(self, dataset: List[Dict]) -> Dict:

        metrics = {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "context_f1": 0.0,
            "answer_survival_rate": 0.0,
            "total_queries": len(dataset),
        }

        total_orig_chars = 0
        total_comp_chars = 0
        start_time = time.time()

        for item in tqdm(dataset, desc="Evaluating Compressor"):

            query = item["question"]
            short_answer = item["answer"]

            docs = []
            gold_sentences = []

            # Build context map
            context_map = {title: sentences for title, sentences in item["context"]}

            # Build SearchResult docs
            for i, (title, sentences) in enumerate(item["context"]):
                full_doc_text = " ".join(sentences)

                docs.append(
                    SearchResult(
                        evi_id=i,
                        docid=i,
                        title=title,
                        text=full_doc_text,
                    )
                )

            # Extract gold supporting sentences
            for title, sent_idx in item["supporting_facts"]:
                if title in context_map and sent_idx < len(context_map[title]):
                    gold_sentences.append(context_map[title][sent_idx])

            gold_text = " ".join(gold_sentences)

            orig_text = " ".join([d.text for d in docs])
            total_orig_chars += len(orig_text)

          
            result = self.compressor.compress(query, docs)

            # Expect compressors to return dict with compressed_docs
            compressed_docs = result["compressed_docs"]

            comp_text = " ".join([d.text for d in compressed_docs])
            total_comp_chars += len(comp_text)

           
            scores = context_overlap_scores(comp_text, gold_text)

            metrics["context_precision"] += scores["precision"]
            metrics["context_recall"] += scores["recall"]
            metrics["context_f1"] += scores["f1"]

            if has_answer([short_answer], comp_text, self.tokenizer):
                metrics["answer_survival_rate"] += 1

        n = metrics["total_queries"]

        if n > 0:
            metrics["context_precision"] = (metrics["context_precision"] / n) * 100
            metrics["context_recall"] = (metrics["context_recall"] / n) * 100
            metrics["context_f1"] = (metrics["context_f1"] / n) * 100
            metrics["answer_survival_rate"] = (
                metrics["answer_survival_rate"] / n
            ) * 100

        if total_orig_chars > 0:
            metrics["compression_ratio_chars"] = (
                1 - (total_comp_chars / total_orig_chars)
            ) * 100
        else:
            metrics["compression_ratio_chars"] = 0.0

        metrics["avg_latency_sec"] = (
            (time.time() - start_time) / n if n > 0 else 0.0
        )

        return metrics