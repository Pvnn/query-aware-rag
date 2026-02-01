import torch
import numpy as np
import re
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
  
  Integrates the official QUITO logic:
  1. Cross-Attention Extraction
  2. Subword Reconstruction (Merge tokens -> Words)
  3. Gaussian Smoothing
  4. Sentence-level Filtering (Max-Word Strategy)
  """
  
  def __init__(self, model_name: str = "google/flan-t5-small", device: str = None):
    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device
        
    print(f"Loading QUITO-X model ({model_name}) on device: {self.device.upper()}")
    
    self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    self.model = T5ForConditionalGeneration.from_pretrained(
      model_name, 
      output_attentions=True
    ).to(self.device)
    self.model.eval()

  def _softmax(self, x, axis=0):
    """Normalize scores to probability distribution."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

  def _reconstruct(self, tokens: List[str], attention: np.ndarray):
    """
    Refined version of the 'reconstruct' util.
    Merges subwords and aggregates attention scores.
    """
    reconstructed_words = []
    reconstructed_attn = []
    
    current_word = ""
    current_attn = 0.0
    
    for token, attn in zip(tokens, attention):
      # T5 uses ' ' (U+2581) as a space marker.
      # Handle standard T5 tokens and potentially others.
      is_start = token.startswith(" ") or token.startswith("Ġ")
      
      clean_token = token.replace(" ", "").replace("Ġ", "").replace("Ċ", "")
      
      if not clean_token:
        continue

      if is_start and current_word:
        # Save the previous word
        reconstructed_words.append(current_word)
        reconstructed_attn.append(current_attn)
        
        # Start new
        current_word = clean_token
        current_attn = attn
      else:
        # Merge subword
        if not current_word: # Edge case: first token
          current_word = clean_token
          current_attn = attn
        else:
          current_word += clean_token
          # QUITO Logic: 'max' score is better for peak detection than sum
          current_attn = max(current_attn, attn)
    
    # Append final word
    if current_word:
      reconstructed_words.append(current_word)
      reconstructed_attn.append(current_attn)
        
    return reconstructed_words, np.array(reconstructed_attn)

  def _get_window_scores(self, query: str, text: str) -> Tuple[List[str], np.ndarray]:
    """
    Processes text in windows to handle length > 512 tokens.
    Returns aggregated WORDS and their SMOOTHED SCORES.
    """
    # 1. Tokenize entire text and query
    text_ids = self.tokenizer.encode(text, add_special_tokens=False)
    query_ids = self.tokenizer.encode(query, add_special_tokens=False)
    
    # Max tokens for text context (leave room for Query + EOS)
    max_chunk = 510 - len(query_ids)
    
    all_words = []
    all_scores = []
    
    # 2. Process in Chunks (Sliding Window)
    for i in range(0, len(text_ids), max_chunk):
      chunk_ids = text_ids[i : i + max_chunk]
      
      # Prepare Input: <Chunk> <Query> <EOS>
      input_ids = chunk_ids + query_ids + [self.tokenizer.eos_token_id]
      input_tensor = torch.tensor([input_ids]).to(self.device)
      decoder_input = torch.tensor([[self.model.config.decoder_start_token_id]]).to(self.device)
      
      # Forward Pass
      with torch.no_grad():
        outputs = self.model(input_ids=input_tensor, decoder_input_ids=decoder_input)
    
      # Extract Attention (Last Layer, Head Avg, First Dec Token)
      # Shape: [seq_len]
      attentions = outputs.cross_attentions[-1][0, :, 0, :].mean(dim=0).cpu().numpy()
      
      # Isolate Chunk Attention
      chunk_attn = attentions[:len(chunk_ids)]
      chunk_tokens = self.tokenizer.convert_ids_to_tokens(chunk_ids)
      
      # Reconstruct Words for this chunk
      words, scores = self._reconstruct(chunk_tokens, chunk_attn)
      
      all_words.extend(words)
      all_scores.extend(scores)

    return all_words, np.array(all_scores)

  def compress(self, query: str, sentences: List[str], compression_ratio: float) -> List[str]:
    """
    Main filtering pipeline.
    """
    if not sentences:
      return []

    # 1. Join sentences to preserve global flow (needed for proper tokenization reconstruction)
    # We use a special separator that T5 treats as a space or punctuation to avoid merging words across sentences.
    full_text = " ".join(sentences)
    
    # 2. Get Word-Level Scores (with Windowing protection)
    words, scores = self._get_window_scores(query, full_text)
    
    # 3. Apply Gaussian Smoothing (from utils)
    # Sigma=1.0 spreads importance to immediate neighbors
    if len(scores) > 0:
      scores = gaussian_filter1d(scores, sigma=1.0)
        
    # 4. Map Word Scores back to Sentences
    # Instead of using NLTK to re-split (which is unreliable if sentences contain dots),
    # we try to align the word stream to the sentence list.
    
    scored_sentences = []
    word_cursor = 0
    
    for sent in sentences:
      # Approximate how many words are in this sentence
      # We clean the sentence similar to the 'reconstruct' logic to match word counts roughly
      sent_clean_words = [w for w in re.split(r'\W+', sent) if w]
      count = len(sent_clean_words)
      
      # Fallback: if regex failed to find words, assume at least 1 to advance cursor
      if count == 0: count = 1
      
      # Slice scores
      end_cursor = min(word_cursor + count, len(scores))
      segment_scores = scores[word_cursor : end_cursor]
      
      # Score Strategy: MAX (Peak Detection)
      if len(segment_scores) > 0:
        sent_score = np.max(segment_scores)
      else:
        sent_score = 0.0
      
      scored_sentences.append((sent, sent_score))
      
      # Advance cursor
      word_cursor = end_cursor

    # 5. Filter (Top-K)
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    num_keep = max(1, int(len(sentences) * compression_ratio))
    
    top_sentences_set = set(sent for sent, score in scored_sentences[:num_keep])
    
    # Return in original order
    return [s for s in sentences if s in top_sentences_set]