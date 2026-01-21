import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.retriever import DenseRetriever

def test_retrieval():
    print("\n" + "="*60)
    print("Test: Dense Retrieval")
    print("="*60)

    # Sample corpus
    documents = [
        "SSDs use flash memory for fast data access.",
        "Hard disk drives have mechanical spinning platters.",
        "Python is a programming language.",
        "Machine learning uses statistical models.",
        "SSDs are faster than HDDs for storage."
    ]

    retriever = DenseRetriever()
    retriever.index_documents(documents)

    query = "How do SSDs work?"
    results = retriever.retrieve(query, top_k=3)

    print(f"\nQuery: {query}\n")
    print("Top-3 Retrieved Documents:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [Score: {score:.3f}] {doc['text']}")

    # Verify SSD-related docs are top ranked
    assert "SSD" in results[0][0]['text']
    print("\n✓ Retrieval test passed")

if __name__ == "__main__":
    test_retrieval()
