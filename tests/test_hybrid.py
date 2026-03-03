import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.compression.hybrid_compressor import HybridCompressor
import os
token = os.getenv("HF_TOKEN") 

def test_pipeline():
  print("🚀 Starting Hybrid Pipeline Test...")
  compressor = HybridCompressor(exit_token=token)
  
  # Mock Data: Mix of World Cup facts, fruit facts, and coding facts
  query = "Who won the 1998 World Cup?"
  
  text = (
    "The 1998 FIFA World Cup was the 16th FIFA World Cup. "
    "It was held in France from 10 June to 12 July 1998. "
    "The country of France was chosen as the host nation by FIFA for the second time. "
    "Bananas are rich in potassium and are a popular fruit. "
    "The tournament was won by France, who beat Brazil 3-0 in the final. "
    "Python is a high-level, general-purpose programming language. "
    "Zinedine Zidane scored two goals in the final match. "
    "The weather in Paris is usually mild in the summer."
  )
  
  print(f"\nQuery: {query}")
  print(f"Original Text Length: {len(text)} chars")
  
  # Run Compression
  # QUITO-X: Keep top 70%
  # EXIT: Threshold 0.5
  result = compressor.compress(
    query=query, 
    context=text, 
    coarse_ratio=0.7, 
    fine_threshold=0.5
  )
  
  metrics = result["metrics"]
  data = result["data"]
  
  print("\n" + "="*50)
  print("📊 PIPELINE DEBUG REPORT")
  print("="*50)
  
  print(f"\n--- 1. Original Input ({metrics['original_count']} sentences) ---")
  # We re-split just to show what the model saw
  for i, s in enumerate(compressor._split_sentences(text)):
    print(f"[{i}] {s}")

  print(f"\n--- 2. Stage 3: QUITO-X Output ({metrics['stage3_count']} kept) ---")
  for i, s in enumerate(data["stage3_output"]):
    print(f"[{i}] {s}")
    
  print(f"\n--- 3. Stage 4: EXIT Output ({metrics['final_count']} kept) ---")
  if metrics['final_count'] == 0:
    print("⚠️  EXIT dropped ALL sentences!")
    print("    Let's check the scores EXIT assigned:")
    for item in data["exit_scores"]:
      print(f"    - Score: {item['score']:.4f} | Sent: {item['sentence'][:50]}...")
  else:
    for i, s in enumerate(data["stage4_output"]):
      print(f"[{i}] {s}")

  print("\n" + "-" * 50)
  print(f"Final Text:\n{result['final_text']}")
  print("="*50)

if __name__ == "__main__":
  test_pipeline()