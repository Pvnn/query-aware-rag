import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

token = os.getenv("HF_TOKEN")
from src.compression.exit_baseline import ExitBaselineCompressor

def test_sentence_decomposition():
	print("Test 1: Sentence Decomposition")
	
	compressor = ExitBaselineCompressor(token=token)
	
	text = "SSDs use flash memory. They are faster than HDDs. Modern computers use them."
	sentences = compressor.decompose_sentences(text)
	
	print(f"Input text: {text}")
	print(f"Extracted sentences: {sentences}")
	print(f"Number of sentences: {len(sentences)}")
	
	assert len(sentences) == 3, f"Expected 3 sentences, got {len(sentences)}"
	print("✓ Sentence decomposition test passed\n")

def test_compression():
	print("Test 2: Document Compression")
	
	compressor = ExitBaselineCompressor(token=token)
	
	query = "How do SSDs improve performance?"
	document = """SSDs use flash memory to store data.
I bought my laptop yesterday.
They have faster read/write speeds than HDDs.
The weather is nice today."""
	
	print(f"Query: {query}")
	print(f"Original document:\n{document}\n")
	
	result = compressor.compress_with_stats(query, document)
	
	print(f"Compressed document:\n{result['compressed_text']}\n")
	print(f"Original length: {result['original_length']} chars")
	print(f"Compressed length: {result['compressed_length']} chars")
	print(f"Compression ratio: {result['compression_ratio']:.2%}")
	print(f"Sentences kept: {result['sentences_kept']}/{result['sentences_total']}")
	
	print("\nSentence-level analysis:")
	for i, score_info in enumerate(result['sentence_scores'], 1):
		print(f"  {i}. [{score_info['score']:.3f}] {'✓' if score_info['kept'] else '✗'} {score_info['sentence']}")
	
	assert len(result['compressed_text']) < len(document), "Compression failed"
	assert len(result['compressed_text']) > 0, "Compressed text is empty"
	print("\n✓ Compression test passed\n")

def test_multiple_queries():
	print("Test 3: Multiple Query Scenarios")
	
	compressor = ExitBaselineCompressor(token=token)
	
	test_cases = [
		{
			"query": "What is machine learning?",
			"document": "Machine learning is a subset of AI. It uses statistical techniques. The weather is sunny today."
		},
		{
			"query": "How does photosynthesis work?",
			"document": "Plants convert sunlight into energy. This process is called photosynthesis. I like pizza."
		}
	]
	
	for i, case in enumerate(test_cases, 1):
		print(f"\nCase {i}:")
		print(f"Query: {case['query']}")
		
		result = compressor.compress_with_stats(case['query'], case['document'])
		
		print(f"Original: {case['document']}")
		print(f"Compressed: {result['compressed_text']}")
		print(f"Ratio: {result['compression_ratio']:.2%}")
		print(f"Kept: {result['sentences_kept']}/{result['sentences_total']} sentences")
	
	print("\n✓ Multiple query test passed\n")

if __name__ == "__main__":
	print("EXIT Baseline Compressor Test Suite")
	
	test_sentence_decomposition()
	test_compression()
	test_multiple_queries()
	
	print("All tests completed successfully! ✓")
