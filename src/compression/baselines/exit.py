"""
Compressor adapted from the EXIT repository.

Repository:
https://github.com/ThisIsHwang/EXIT

Paper:
EXIT: Extractive Context Compression for Retrieval-Augmented Generation
Zheng et al., 2024
"""

import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from functools import lru_cache

from src.eval.interfaces import BaseCompressor, SearchResult


class EXITCompressor(BaseCompressor):
    """EXIT: Context-aware extractive compression."""
    
    def __init__(
        self,
        token: str = None,
        base_model: str = "google/gemma-2b-it",
        checkpoint: str = None,
        device: str = None,
        cache_dir: str = "./cache",
        batch_size: int = 8,
        threshold: float = 0.5
    ):
        self.batch_size = batch_size
        self.threshold = threshold
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing official EXITCompressor with {base_model}...")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # FIX: Resolves load_in_4bit deprecation warnings
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=self.device,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            token=token
        )
        
        if checkpoint:
            self.peft_config = PeftConfig.from_pretrained(checkpoint, token=token)
            self.model = PeftModel.from_pretrained(
                self.base_model,
                checkpoint,
                token=token
            )
        else:
            self.model = self.base_model
            
        self.model.eval()
            
        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        
        torch.cuda.empty_cache()
    
    @lru_cache(maxsize=1024)
    def _generate_prompt(self, query: str, context: str, sentence: str) -> str:
        return (
            f'<start_of_turn>user\n'
            f'Query:\n{query}\n'
            f'Full context:\n{context}\n'
            f'Sentence:\n{sentence}\n'
            f'Is this sentence useful in answering the query? '
            f'Answer only "Yes" or "No".<end_of_turn>\n'
            f'<start_of_turn>model\n'
        )
    
    def _predict_batch(self, queries: List[str], contexts: List[str], sentences: List[str]) -> List[float]:
        prompts = [
            self._generate_prompt(q, c, s)
            for q, c, s in zip(queries, contexts, sentences)
        ]
        
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=4096,
            return_attention_mask=True
        )
        
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        
        with torch.no_grad():
            # FIX: Modern autocast syntax removes deprecation warning
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = self.model(**inputs)
                
                next_token_logits = outputs.logits[:, -1, :]
                relevant_logits = torch.stack([
                    next_token_logits[:, self.yes_token_id],
                    next_token_logits[:, self.no_token_id]
                ], dim=1)
                
                probs = torch.softmax(relevant_logits, dim=1)
                
                # Extract just the "Yes" probability for thresholding
                yes_probs = probs[:, 0].cpu().tolist()
                
        return yes_probs
    
    def compress(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        context = "\n".join(f"{doc.title}\n{doc.text}" for doc in documents)
        
        selected_texts = []
        current_doc_id = None
        current_texts = []
        
        # FIX: We now batch the documents so the GPU actually runs at full speed
        doc_texts = [doc.text for doc in documents]
        probs_all = []
        
        for i in range(0, len(doc_texts), self.batch_size):
            batch_texts = doc_texts[i : i + self.batch_size]
            queries = [query] * len(batch_texts)
            contexts = [context] * len(batch_texts)
            
            probs = self._predict_batch(queries, contexts, batch_texts)
            probs_all.extend(probs)
        
        # Reconstruct the original logic with the batch-computed probabilities
        for doc, prob in zip(documents, probs_all):
            if current_doc_id != doc.evi_id:
                if current_texts:
                    doc_text = " ".join(current_texts)
                    if doc_text.strip():
                        selected_texts.append(doc_text)
                current_doc_id = doc.evi_id
                current_texts = []
            
            if prob >= self.threshold:
                current_texts.append(doc.text)
        
        if current_texts:
            doc_text = " ".join(current_texts)
            if doc_text.strip():
                selected_texts.append(doc_text)
        
        compressed_text = "\n\n".join(selected_texts)
        
        return [SearchResult(
            evi_id=0,
            docid=0,
            title="EXIT",
            text=compressed_text,
            score=1.0
        )]