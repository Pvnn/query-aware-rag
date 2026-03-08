import sys
import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import List

from src.eval.interfaces import SearchResult, BaseCompressor
from src.eval.eval_pipeline import CompressorEvaluator

from src.compression.hybrid_compressor import HybridCompressor

# --- 1. THE ADAPTER ---
class HybridCompressorAdapter(BaseCompressor):
  """
  Wraps the complex HybridCompressor to fit the simple BaseCompressor interface.
  """
  def __init__(self, compressor: HybridCompressor):
    self.compressor = compressor

  def compress(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
    compressed_docs = []
    
    for doc in documents:
      # Call our actual compressor
      result = self.compressor.compress(
        query=query,
        context=doc.text,
        coarse_ratio=0.7,
        fine_threshold=0.5,
        use_coarse=True,
        use_fine=True
      )
      
      # Extract just the compressed text and package it back into a SearchResult
      compressed_docs.append(SearchResult(
        evi_id=doc.evi_id,
        docid=doc.docid,
        title=doc.title,
        text=result['final_text']
      ))
      
    return compressed_docs


# --- 2. THE TEST EXECUTION ---
def test_real_pipeline_eval():
  print("Loading Environment and Models... (This will be much faster now!)")
  load_dotenv()
  token = os.getenv("HF_TOKEN")
  
  # Initialize actual model (NO LLM READER NEEDED)
  real_compressor = HybridCompressor(exit_token=token)
  
  # Wrap the compressor
  adapter = HybridCompressorAdapter(real_compressor)
  
  # Initialize the NEW Extractive Evaluator
  evaluator = CompressorEvaluator(compressor=adapter)
  
  # A realistic dummy dataset item based on the HotpotQA format
  dummy_data = [
    {
      "question": "What are the specific instruments on the James Webb Space Telescope used for observation?",
      "answer": "Near-Infrared Camera",
      "context": [
        [
          "JWST", 
          [
            "The James Webb Space Telescope (JWST) is a space telescope designed to conduct infrared astronomy.",
            "Its high-resolution and high-sensitivity instruments allow it to view objects too old, distant, or faint for the Hubble Space Telescope.",
            "The primary scientific instruments on board include the Near-Infrared Camera (NIRCam) and the Mid-Infrared Instrument (MIRI).",
            "These tools are essential for studying the formation of early galaxies.",
            "Additionally, the telescope requires a massive sunshield to keep its instruments cold."
          ]
        ]
      ],
      "supporting_facts": [
        ["JWST", 1], # The sentence providing the context about instruments
        ["JWST", 2]  # The sentence providing the exact answer
      ]
    }
  ]
  
  print("\n--- Running Extractive Evaluation on Real Compressor ---")
  results = evaluator.evaluate(dummy_data)
  
  print("\n📊 Extractive Evaluation Results:")
  print(f"  Context Precision:    {results['context_precision']:.1f}%")
  print(f"  Context Recall:       {results['context_recall']:.1f}%")
  print(f"  Context F1:           {results['context_f1']:.1f}%")
  print(f"  Answer Survival Rate: {results['answer_survival_rate']:.1f}%")
  print(f"  Compression Ratio:    {results['compression_ratio_chars']:.1f}%")
  print(f"  Avg Latency:          {results['avg_latency_sec']:.2f}s")

if __name__ == "__main__":
  test_real_pipeline_eval()