"""
LongLLMLingua Compressor Adapter

Adapted from:
EXIT repository
https://github.com/ThisIsHwang/EXIT
"""

from typing import List
from llmlingua import PromptCompressor

from src.eval.interfaces import BaseCompressor, SearchResult


class LongLLMLinguaCompressor(BaseCompressor):
    """
    LongLLMLingua: token-level context compression.
    """

    def __init__(
        self,
        device_map: str = "auto",
        cache_dir: str = "./cache",
        compression_ratio: float = 0.4,
        context_budget: str = "+100",
        compression_rate: float = 0.2,
        use_sentence_level: bool = False,
        reorder_context: str = "sort",
    ):

        self.device_map = device_map
        self.compression_ratio = compression_ratio
        self.context_budget = context_budget
        self.compression_rate = compression_rate
        self.use_sentence_level = use_sentence_level
        self.reorder_context = reorder_context

        self.compressor = PromptCompressor(
            device_map=device_map
        )

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:

        prompt_list = [
            f"{doc.title}\n{doc.text}"
            for doc in documents
        ]

        result = self.compressor.compress_prompt(
            prompt_list,
            instruction=(
                "Answer the query using only the context."
            ),
            question=query,
            rate=self.compression_rate,
            condition_compare=True,
            condition_in_question="after",
            rank_method="longllmlingua",
            use_sentence_level_filter=self.use_sentence_level,
            context_budget=self.context_budget,
            dynamic_context_compression_ratio=self.compression_ratio,
            reorder_context=self.reorder_context,
        )

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="longllmlingua",
                text=result["compressed_prompt"],
                score=1.0,
            )
        ]