import torch
import numpy as np
from typing import List, Tuple
from transformers import T5Tokenizer, T5ForConditionalGeneration
from scipy.ndimage import gaussian_filter1d

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class QuitoxCoarseFilter:
    """
    Module 3: QUITO-X Coarse Filter (Min-Max Normalized Dynamic Thresholding)
    """
    
    def __init__(self, model_name: str = "google/flan-t5-small", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading QUITO-X model ({model_name}) on device: {self.device.upper()}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _softmax(self, x, axis=0):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def _get_sentence_scores(self, query: str, sentences: List[str]) -> Tuple[List[float], int]:
        query_ids = self.tokenizer.encode(query, add_special_tokens=False)
        
        flat_ids = []
        boundaries = []
        
        for idx, sent in enumerate(sentences):
            prefix = " " if idx > 0 and not sent.startswith(" ") else ""
            ids = self.tokenizer.encode(prefix + sent, add_special_tokens=False)
            
            start_idx = len(flat_ids)
            end_idx = start_idx + len(ids)
            
            boundaries.append((start_idx, end_idx))
            flat_ids.extend(ids)

        max_chunk = 510 - len(query_ids)
        token_attentions = np.zeros(len(flat_ids))
        total_tokens_consumed = 0
        
        for i in range(0, len(flat_ids), max_chunk):
            chunk_ids = flat_ids[i : i + max_chunk]
            input_ids = chunk_ids + query_ids + [self.tokenizer.eos_token_id]
            total_tokens_consumed += len(input_ids)
            
            input_tensor = torch.tensor([input_ids]).to(self.device)
            decoder_input = torch.tensor([[self.model.config.decoder_start_token_id]]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_tensor, decoder_input_ids=decoder_input, output_attentions=True)
        
            attentions = outputs.cross_attentions[-1][0, :, 0, :].mean(dim=0).cpu().numpy()
            chunk_attn = attentions[:len(chunk_ids)]
            token_attentions[i : i + len(chunk_ids)] = chunk_attn

        if len(token_attentions) > 0:
            smoothed_attentions = gaussian_filter1d(token_attentions, sigma=2.0)
        else:
            smoothed_attentions = token_attentions

        sent_scores = []
        for start, end in boundaries:
            segment = smoothed_attentions[start:end]
            if len(segment) > 0:
                score = np.max(segment)
            else:
                score = 0.0
            sent_scores.append(float(score))

        return sent_scores, total_tokens_consumed

    def compress(self, query: str, sentences: List[str], tolerance_ratio: float = 0.6, min_keep: int = 2) -> dict:
        """
        Compresses sentences dynamically using Min-Max Normalized Attention Scores.
        Removes the hard ceiling so ALL highly relevant sentences are kept.
        """
        if not sentences:
            return {"filtered_sentences": [], "total_tokens_consumed": 0, "quitox_details": []}

        # 1. Get raw softmax scores
        raw_scores, total_tokens = self._get_sentence_scores(query, sentences)
        
        # 2. Apply Min-Max Normalization to stretch scores between 0.0 and 1.0
        if len(raw_scores) > 1:
            min_score = min(raw_scores)
            max_score = max(raw_scores)
            if max_score > min_score:
                normalized_scores = [(s - min_score) / (max_score - min_score) for s in raw_scores]
            else:
                normalized_scores = [1.0 for _ in raw_scores]
        else:
            normalized_scores = [1.0 for _ in raw_scores]

        # 3. Pair sentences with their new normalized scores
        scored_sentences = [
            {"text": s, "raw_score": raw, "norm_score": norm} 
            for s, raw, norm in zip(sentences, raw_scores, normalized_scores)
        ]

        # 4. Filter logic (No more hard ceiling!)
        keep_threshold = tolerance_ratio 

        filtered_sentences = []
        quitox_details = []
        
        # Sort by score purely to enforce the minimum keep guarantee
        sorted_indices = sorted(range(len(scored_sentences)), key=lambda i: scored_sentences[i]['norm_score'], reverse=True)
        
        indices_to_keep = set()
        for rank, idx in enumerate(sorted_indices):
            # Keep it if it meets the high-attention threshold OR if we haven't satisfied the minimum safety net
            if scored_sentences[idx]['norm_score'] >= keep_threshold or rank < min_keep:
                indices_to_keep.add(idx)

        # 5. Reconstruct exactly in original document order
        for idx, s in enumerate(scored_sentences):
            is_retained = idx in indices_to_keep
            if is_retained:
                filtered_sentences.append(s['text'])
            
            quitox_details.append({
                "text": s['text'],
                "score": round(s['norm_score'], 3), 
                "raw_score": round(s['raw_score'], 5),
                "retained": is_retained
            })

        return {
            "filtered_sentences": filtered_sentences,
            "kept_indices": sorted(indices_to_keep),  
            "total_tokens_consumed": total_tokens,
            "quitox_details": quitox_details
        }