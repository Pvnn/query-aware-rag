import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.retriever import DenseRetriever
from src.compression.exit_baseline import ExitBaselineCompressor
from src.generation.reader import RAGReader
import time

class QueryAwareRAG:
  def __init__(self, token):
    print("Initializing Query-Aware RAG Pipeline...")
    self.retriever = DenseRetriever()
    self.compressor = ExitBaselineCompressor(token=token)
    self.reader = RAGReader()
    print("✓ Pipeline initialized\n")
  
  def run(self, query, top_k=5):
    """Run end-to-end RAG pipeline."""
    print(f"Query: {query}\n")
    
    # Step 1: Retrieval
    print(f"[1/3] Retrieving top-{top_k} documents...")
    start = time.time()
    retrieved_docs = self.retriever.retrieve(query, top_k)
    retrieval_time = time.time() - start
    print(f"  ✓ Retrieved {len(retrieved_docs)} docs in {retrieval_time:.2f}s\n")
    
    # Step 2: Compression
    print(f"[2/3] Compressing documents...")
    start = time.time()
    compressed_docs = []
    original_tokens = 0
    compressed_tokens = 0
    
    for doc, score in retrieved_docs:
      original_text = doc['text']
      compressed_text = self.compressor.compress(query, original_text)
      compressed_docs.append(compressed_text)
      
      # Rough token count (words * 1.3)
      original_tokens += len(original_text.split()) * 1.3
      compressed_tokens += len(compressed_text.split()) * 1.3
    
    context = " ".join(compressed_docs)
    compression_time = time.time() - start
    compression_ratio = (1 - compressed_tokens/original_tokens) * 100 if original_tokens > 0 else 0
    print(f"  ✓ Compressed in {compression_time:.2f}s")
    print(f"  ✓ Compression ratio: {compression_ratio:.1f}% ({int(original_tokens)} → {int(compressed_tokens)} tokens)\n")
    
    # Step 3: Generation
    print(f"[3/3] Generating answer...")
    start = time.time()
    answer = self.reader.generate_answer(query, context)
    generation_time = time.time() - start
    print(f"  ✓ Generated in {generation_time:.2f}s\n")
    
    # Results
    total_time = retrieval_time + compression_time + generation_time
    print(f"Answer: {answer}")
    print(f"\nTotal time: {total_time:.2f}s")
    
    return {
      'query': query,
      'answer': answer,
      'context': context,
      'retrieved_docs': retrieved_docs,
      'metrics': {
        'retrieval_time': retrieval_time,
        'compression_time': compression_time,
        'generation_time': generation_time,
        'total_time': total_time,
        'compression_ratio': compression_ratio,
        'original_tokens': int(original_tokens),
        'compressed_tokens': int(compressed_tokens)
      }
    }
