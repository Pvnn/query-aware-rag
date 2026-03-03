import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.retriever import DenseRetriever
from src.compression.hybrid_compressor import HybridCompressor
from src.generation.reader import RAGReader
import time

class QueryAwareRAG:
  def __init__(self, token, use_coarse=True, use_fine=True):
    print("Initializing Query-Aware RAG Pipeline...")
    
    # Pipeline Components
    self.retriever = DenseRetriever()
    
    # Using Hybrid Compressor
    self.compressor = HybridCompressor(exit_token=token)
    
    # Configuration flags
    self.use_coarse = use_coarse
    self.use_fine = use_fine
    
    self.reader = RAGReader()
    
    print(f"Config: Coarse={self.use_coarse}, Fine={self.use_fine}")
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
    
    # Process each document individually through the Hybrid Pipeline
    for i, (doc, score) in enumerate(retrieved_docs):
      original_text = doc['text']
      
      # Run Hybrid Compression
      result = self.compressor.compress(
        query=query, 
        context=original_text,
        coarse_ratio=0.7,      # Aggressive coarse filtering
        fine_threshold=0.5,    # Standard fine filtering
        use_coarse=self.use_coarse,
        use_fine=self.use_fine
      )
      
      compressed_text = result['final_text']
      compressed_docs.append(compressed_text)
      
      # Stats logging
      input_len = result['metrics']['original_count']
      s3_len = result['metrics']['stage3_count']
      final_len = result['metrics']['final_count']
      
      # Rough token count (words * 1.3)
      orig_tok = len(original_text.split()) * 1.3
      comp_tok = len(compressed_text.split()) * 1.3
      
      original_tokens += orig_tok
      compressed_tokens += comp_tok
      
      print(f"  Doc {i+1}: {input_len} sents -> {s3_len} (Coarse) -> {final_len} (Fine)")
    
    context = " ".join(compressed_docs)
    compression_time = time.time() - start
    
    if original_tokens > 0:
      compression_ratio = (1 - compressed_tokens/original_tokens) * 100 
    else:
      compression_ratio = 0
      
    print(f"  ✓ Compressed in {compression_time:.2f}s")
    print(f"  ✓ Total Ratio: {compression_ratio:.1f}% ({int(original_tokens)} → {int(compressed_tokens)} tokens)\n")
    
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