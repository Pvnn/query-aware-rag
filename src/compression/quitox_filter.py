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
  Module 3: QUITO-X Coarse Filter
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
    """
    Tokenize each sentence to get its exact token length.
    Then map the raw token attentions back using these exact index boundaries.
    """
    query_ids = self.tokenizer.encode(query, add_special_tokens=False)
    
    # 1. Track precise token boundaries for each sentence
    flat_ids = []
    boundaries = []
    
    for idx, sent in enumerate(sentences):
      # Add space prefix to simulate continuous text for the T5 tokenizer
      prefix = " " if idx > 0 and not sent.startswith(" ") else ""
      ids = self.tokenizer.encode(prefix + sent, add_special_tokens=False)
      
      start_idx = len(flat_ids)
      end_idx = start_idx + len(ids)
      
      boundaries.append((start_idx, end_idx))
      flat_ids.extend(ids)

    max_chunk = 510 - len(query_ids)
    
    # Array to hold the attention score for every single token
    token_attentions = np.zeros(len(flat_ids))
    total_tokens_consumed = 0
    
    # 2. Chunk processing
    for i in range(0, len(flat_ids), max_chunk):
      chunk_ids = flat_ids[i : i + max_chunk]
      input_ids = chunk_ids + query_ids + [self.tokenizer.eos_token_id]
      total_tokens_consumed += len(input_ids)
      
      input_tensor = torch.tensor([input_ids]).to(self.device)
      decoder_input = torch.tensor([[self.model.config.decoder_start_token_id]]).to(self.device)
      
      with torch.no_grad():
        outputs = self.model(input_ids=input_tensor, decoder_input_ids=decoder_input, output_attentions=True)
    
      # Extract cross-attention for the chunk
      attentions = outputs.cross_attentions[-1][0, :, 0, :].mean(dim=0).cpu().numpy()
      chunk_attn = attentions[:len(chunk_ids)]
      
      # Map directly back to the global token array
      token_attentions[i : i + len(chunk_ids)] = chunk_attn

    # 3. Apply smoothing to the contiguous token array
    if len(token_attentions) > 0:
      # Increased sigma slightly because we are smoothing over tokens, not words
      smoothed_attentions = gaussian_filter1d(token_attentions, sigma=2.0)
    else:
      smoothed_attentions = token_attentions

    # 4. Extract Max token-score per sentence using our exact boundaries
    sent_scores = []
    for start, end in boundaries:
      segment = smoothed_attentions[start:end]
      if len(segment) > 0:
        score = np.max(segment)
      else:
        score = 0.0
      sent_scores.append(float(score))

    return sent_scores, total_tokens_consumed

  def compress(self, query: str, sentences: List[str], compression_ratio: float) -> dict:
    if not sentences:
      return {"filtered_sentences": [], "total_tokens_consumed": 0, "quitox_details": []}

    # Get exact scores using token boundaries
    sent_scores, total_tokens = self._get_sentence_scores(query, sentences)
    
    scored_sentences = [{"text": s, "score": score} for s, score in zip(sentences, sent_scores)]

    # Determine Top-K threshold
    sorted_scores = sorted([s['score'] for s in scored_sentences], reverse=True)
    num_keep = max(1, int(len(sentences) * compression_ratio))
    
    threshold_idx = min(num_keep - 1, len(sorted_scores) - 1)
    keep_threshold = sorted_scores[threshold_idx] if sorted_scores else 0.0

    filtered_sentences = []
    quitox_details = []
    
    for s in scored_sentences:
      is_retained = s['score'] >= keep_threshold and len(filtered_sentences) < num_keep
      if is_retained:
        filtered_sentences.append(s['text'])
      
      quitox_details.append({
        "text": s['text'],
        "score": round(s['score'], 4),
        "retained": is_retained
      })

    return {
      "filtered_sentences": filtered_sentences,
      "total_tokens_consumed": total_tokens,
      "quitox_details": quitox_details
    }