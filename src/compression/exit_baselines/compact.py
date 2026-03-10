"""
CompAct Compressor Adapter

Adapted from:
EXIT repository
https://github.com/ThisIsHwang/EXIT

Paper:
EXIT: Extractive Context Compression for Retrieval-Augmented Generation
"""

import re
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.eval.interfaces import BaseCompressor, SearchResult


class CompactCompressor(BaseCompressor):
    """
    CompAct: Context compression using iterative summarization.
    """

    def __init__(
        self,
        model_dir: str = "cwyoon99/CompAct-7b",
        device: str = "cuda",
        cache_dir: str = "./cache",
        batch_size: int = 5,
    ):

        self.device = device
        self.batch_size = batch_size

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
            cache_dir=cache_dir,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True
        )

    def _create_prompt(
        self,
        query: str,
        documents: str,
        prev_summary: str = "",
        prev_eval: str = "",
        iteration: int = 0,
    ) -> str:

        if iteration == 0:
            instruction = (
                "Generate a summary of source documents relevant to the question."
            )

            prompt = f"{instruction}\n\nQuestion: {query}\n\nSource documents: {documents}\n\nSummary:"

        else:
            prompt = (
                f"Question: {query}\n\n"
                f"Previous summary: {prev_summary}\n\n"
                f"Source documents: {documents}\n\nSummary:"
            )

        return prompt

    def _parse_output(self, text: str) -> Dict[str, str]:

        summary_pattern = r"(Summary:)(.*?)(?=Evaluation:|$)"
        evaluation_pattern = r"(Evaluation:)(.*?)(?=Summary:|$)"

        summary_match = re.search(summary_pattern, text, re.DOTALL)
        eval_match = re.search(evaluation_pattern, text, re.DOTALL)

        summary = summary_match.group(2).strip() if summary_match else ""
        evaluation = eval_match.group(2).strip() if eval_match else ""

        return {
            "summary": summary,
            "eval": evaluation,
        }

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:

        prev_summaries = []

        for i in range(0, len(documents), self.batch_size):

            batch_docs = documents[i : i + self.batch_size]

            batch_text = "\n".join(
                f"{doc.title}\n{doc.text}" for doc in batch_docs
            )

            prompt = self._create_prompt(query, batch_text)

            with torch.no_grad():

                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=300,
                    temperature=0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                output_text = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.size(1):],
                    skip_special_tokens=True,
                )

                parsed = self._parse_output(output_text)

                prev_summaries.append(parsed["summary"])

        compressed_text = prev_summaries[-1] if prev_summaries else ""

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="compact",
                text=compressed_text,
                score=1.0,
            )
        ]