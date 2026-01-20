import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import os
token = os.getenv("HF_TOKEN")
from src.rag_pipeline import QueryAwareRAG

def test_pipeline():
  print("Test: Complete RAG Pipeline")
  
  # Sample corpus
  documents = [
    "SSDs use flash memory for fast data access.",
    "Hard disk drives have mechanical spinning platters.",
    "Python is a programming language.",
    "Machine learning uses statistical models.",
    "SSDs are faster than HDDs for storage.",
    "Flash memory has no moving parts.",
    "HDDs are cheaper but slower than SSDs."
  ]
  
  # Initialize pipeline
  pipeline = QueryAwareRAG(token= token)
  
  # Index documents
  pipeline.retriever.index_documents(documents)
  
  # Run query
  query = "How do SSDs work?"
  result = pipeline.run(query, top_k=3)
  
  # Validate
  assert len(result['answer']) > 0
  assert result['metrics']['total_time'] > 0
  print("\n✓ Pipeline test passed")
  print(f"\nMetrics:")
  print(f"  - Total time: {result['metrics']['total_time']:.2f}s")
  print(f"  - Compression: {result['metrics']['compression_ratio']:.1f}%")
  print(f"  - Tokens: {result['metrics']['original_tokens']} → {result['metrics']['compressed_tokens']}")

if __name__ == "__main__":
  test_pipeline()
