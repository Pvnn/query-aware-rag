import torch
import time
import spacy
from typing import List, Union, Dict

from src.compression.quitox_filter import QuitoxCoarseFilter
from src.compression.exit_baseline import ExitBaselineCompressor

class HybridCompressor:
  """
  The Master Module: Adaptive Two-Stage Compression Pipeline.
  
  Flow:
  Input Doc -> [Stage 3: QUITO-X Coarse Filter] -> Coarse Doc -> [Stage 4: EXIT Fine Filter] -> Final Doc
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
    
    # 3. Initialize Stage 4: EXIT
    print("--- [Stage 4] Loading EXIT ---")
    self.exit = ExitBaselineCompressor(
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
    
    # --- Stage 4: EXIT Fine Filter ---
    if use_fine:
      # Prepare context for EXIT
      coarse_context_str = " ".join(current_sentences)
      
      # Update threshold
      self.exit.threshold = fine_threshold
      
      t2 = time.time()
      # Run EXIT
      exit_result = self.exit.compress_with_stats(query, coarse_context_str)
      exit_time = time.time() - t2
      
      final_text = exit_result["compressed_text"]
      
      # Extract kept sentences list for reporting
      if "sentence_scores" in exit_result:
        final_sentences_list = [
          item["sentence"] for item in exit_result["sentence_scores"] 
          if item["kept"]
        ]
      else:
        final_sentences_list = self._split_sentences(final_text)
        
    else:
      # If Fine filter is skipped, the "final" output is just the coarse output
      final_sentences_list = current_sentences
      final_text = " ".join(current_sentences)

    total_time = time.time() - start_time
    
    # --- Reporting ---
    stats = {
      "final_text": final_text,
      "metrics": {
        "original_count": original_count,
        "stage3_count": stage3_count,
        "final_count": len(final_sentences_list),
        "time_quitox": round(quitox_time, 2),
        "time_exit": round(exit_time, 2),
        "time_total": round(total_time, 2)
      }
    }
    
    return stats