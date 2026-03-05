import torch
import time
import spacy
from typing import List, Union, Dict

from src.compression.quitox_filter import QuitoxCoarseFilter
from src.compression.ep_exit import EPExitCompressor 

class HybridCompressor:
  """
  The Master Module: Adaptive Two-Stage Compression Pipeline.
  
  Flow:
  Input Doc -> [Stage 3: QUITO-X Coarse Filter] -> Coarse Doc -> [Stage 4: EP-EXIT Fine Filter] -> Final Doc
  """
  
  def __init__(
    self, 
    exit_token: str,
    quitox_model: str = "google/flan-t5-small",
    exit_model: str = "doubleyyh/exit-gemma-2b",
    device: str = None
  ):
    """
    Args:
      exit_token: HuggingFace token for the EXIT model (Gemma).
      quitox_model: T5 model for coarse filtering (Fast).
      exit_model: Gemma model for fine filtering (Smart).
    """
    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device
      
    print("\n🔗 Initializing HYBRID COMPRESSION PIPELINE...")
    
    # 1. Load Spacy for consistent sentence splitting
    try:
      self.nlp = spacy.load("en_core_web_sm")
    except OSError:
      print("Downloading spacy model...")
      from spacy.cli import download
      download("en_core_web_sm")
      self.nlp = spacy.load("en_core_web_sm")

    # 2. Initialize Stage 3: QUITO-X
    print("--- [Stage 3] Loading QUITO-X ---")
    self.quitox = QuitoxCoarseFilter(model_name=quitox_model, device=self.device)
    
    # 3. Initialize Stage 4: EP-EXIT
    print("--- [Stage 4] Loading EP-EXIT ---")
    self.exit = EPExitCompressor(
      token=exit_token, 
      model_name=exit_model,
      threshold=0.5 # Default threshold
    )
    print("✅ Hybrid Pipeline Ready.\n")

  def _split_sentences(self, text: str) -> List[str]:
    """Helper to ensure robust splitting."""
    doc = self.nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

  def compress(
    self, 
    query: str, 
    context: Union[str, List[str]], 
    coarse_ratio: float = 0.7, 
    fine_threshold: float = 0.5,
    use_coarse: bool = True,
    use_fine: bool = True
  ) -> Dict:
    """
    Runs the pipeline with configurable stages.
    
    Args:
      use_coarse: If False, skips QUITO-X (Stage 3).
      use_fine: If False, skips EXIT (Stage 4).
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    if isinstance(context, str):
      sentences = self._split_sentences(context)
    else:
      sentences = context
      
    original_count = len(sentences)
    current_sentences = sentences
    
    quitox_time = 0.0
    exit_time = 0.0
    
    # --- Stage 3: QUITO-X Coarse Filter ---
    if use_coarse:
      t1 = time.time()
      current_sentences = self.quitox.compress(
        query=query, 
        sentences=current_sentences, 
        compression_ratio=coarse_ratio
      )
      quitox_time = time.time() - t1
    
    stage3_count = len(current_sentences)
    
    # --- Stage 4: EP-EXIT Fine Filter ---
    if use_fine:
      # Prepare context for EP-EXIT
      coarse_context_str = " ".join(current_sentences)
      
      # Update threshold
      self.exit.threshold = fine_threshold
      
      t2 = time.time()
      # Run EP-EXIT and get the enriched stats payload
      exit_result = self.exit.compress_with_stats(query, coarse_context_str)
      exit_time = time.time() - t2
      
      final_text = exit_result["compressed_text"]
      final_sentence_count = exit_result["sentences_kept"]
      
      # Extract our specific evidence unit tracking data
      ep_exit_details = {
        "evidence_units_total": exit_result.get("evidence_units_total", 0),
        "evidence_units_kept_count": exit_result.get("evidence_units_kept_count", 0),
        "evidence_units_removed_count": exit_result.get("evidence_units_removed_count", 0),
        "kept_units": exit_result.get("kept_units", []),
        "removed_units": exit_result.get("removed_units", [])
      }
        
    else:
      # If Fine filter is skipped, the "final" output is just the coarse output
      final_text = " ".join(current_sentences)
      final_sentence_count = len(current_sentences)
      
      # Provide empty tracking data to maintain consistent dictionary structure
      ep_exit_details = {
        "evidence_units_total": 0,
        "evidence_units_kept_count": 0,
        "evidence_units_removed_count": 0,
        "kept_units": [],
        "removed_units": []
      }

    total_time = time.time() - start_time
    
    # --- Final Reporting ---
    stats = {
      "final_text": final_text,
      "metrics": {
        "original_sentence_count": original_count,
        "coarse_sentence_count": stage3_count,
        "final_sentence_count": final_sentence_count,
        "time_quitox": round(quitox_time, 2),
        "time_exit": round(exit_time, 2),
        "time_total": round(total_time, 2)
      },
      "ep_exit_details": ep_exit_details
    }
    
    return stats