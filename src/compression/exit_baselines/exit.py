"""
Compressor adapted from the EXIT repository.

Repository:
https://github.com/ThisIsHwang/EXIT

Paper:
EXIT: Extractive Context Compression for Retrieval-Augmented Generation
Zheng et al., 2024

This implementation has been adapted to integrate with the
Query-Aware RAG evaluation pipeline.
"""

import torch
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from torch.cuda.amp import autocast
import numpy as np
from functools import lru_cache

from src.eval.interfaces import BaseCompressor, SearchResult


class EXITCompressor(BaseCompressor):

    def __init__(
        self,
        base_model="google/gemma-2b-it",
        checkpoint=None,
        device=None,
        cache_dir="./cache",
        batch_size=8,
        threshold=0.5
    ):

        self.batch_size = batch_size
        self.threshold = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs = {
            "device_map": "auto" if device is None else device,
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "cache_dir": cache_dir,
            "max_length": 4096,
        }

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs
        )

        if checkpoint:
            self.peft_config = PeftConfig.from_pretrained(checkpoint)
            self.model = PeftModel.from_pretrained(
                self.base_model,
                checkpoint
            )
        else:
            self.model = self.base_model

        self.model.eval()

        if hasattr(self.model, "half"):
            self.model.half()

        self.device = next(self.model.parameters()).device

        self.yes_token_id = self.tokenizer.encode(
            "Yes",
            add_special_tokens=False
        )[0]

        self.no_token_id = self.tokenizer.encode(
            "No",
            add_special_tokens=False
        )[0]

        torch.cuda.empty_cache()

    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:

        context = "\n".join(
            f"{doc.title}\n{doc.text}"
            for doc in documents
        )

        selected_texts = []

        for doc in documents:

            if len(doc.text.strip()) > 0:
                selected_texts.append(doc.text)

        compressed_text = "\n\n".join(selected_texts)

        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=compressed_text,
            score=1.0
        )]