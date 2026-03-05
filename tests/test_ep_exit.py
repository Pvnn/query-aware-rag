import os
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.compression.ep_exit import EPExitCompressor


def test_ep_exit():
  print("=== Testing EP-EXIT ===\n")
  load_dotenv()
  
  # Get HF token from environment
  token = os.getenv("HF_TOKEN")

  if token is None:
    raise ValueError("HF_TOKEN not set in environment.")

  # Sample document
  document = """
  Insulin is a hormone produced by the pancreas.
  It helps regulate blood sugar levels in the body.
  The Eiffel Tower is located in Paris.
  Diabetes occurs when the body cannot produce enough insulin.
  Insulin therapy is commonly used to treat diabetes.
  Paris is the capital of France.
  """

  query = "How does insulin work in diabetes?"

  compressor = EPExitCompressor(token=token)

  # Call the stats-enriched method
  result = compressor.compress_with_stats(query, document)

  print("Original Document:\n")
  print(document.strip())

  print("\n" + "="*50)
  print("📊 EP-EXIT Compression Stats")
  print("="*50)
  print(f"Original Length: {result['original_length']} chars")
  print(f"Compressed Length: {result['compressed_length']} chars")
  print(f"Compression Ratio: {result['compression_ratio'] * 100:.1f}%")
  print(f"Sentences: {result['sentences_total']} total -> Kept {result['sentences_kept']}")
  print(f"Evidence Units: {result['evidence_units_total']} total -> Kept {result['evidence_units_kept_count']}, Removed {result['evidence_units_removed_count']}")

  print("\n🔍 Evidence Unit Breakdown:")

  if result.get('kept_units'):
    print("\n  ✅ KEPT Units (Passed Threshold):")
    for u in result['kept_units']:
      print(f"     [Score: {u['score']:.4f}] {u['text']}")

  if result.get('removed_units'):
    print("\n  ❌ REMOVED Units (Failed Threshold):")
    for u in result['removed_units']:
      print(f"     [Score: {u['score']:.4f}] {u['text']}")

  print("\n" + "="*50)
  print("Final Compressed Output:\n")
  print(result['compressed_text'])

  print("\n✓ EP-EXIT test completed")


if __name__ == "__main__":
  test_ep_exit()