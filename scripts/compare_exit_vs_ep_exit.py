import sys
from pathlib import Path
import time
import json

# ------------------------------------------------------------------
# Fix imports (project root)
# ------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.retriever import DenseRetriever
from src.compression.exit_baseline import ExitBaselineCompressor
from src.compression.ep_exit import EPExitCompressor

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
QUERY = "What are the symptoms of diabetes?"
TOP_K = 3
EXIT_TOKEN_BUDGET = 200


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------
def approx_tokens(text: str) -> int:
    """Rough token approximation."""
    return int(len(text.split()) * 1.3)


# ------------------------------------------------------------------
# Main comparison logic
# ------------------------------------------------------------------
def main():
    print("=" * 80)
    print("EXIT vs EP-EXIT COMPRESSION COMPARISON")
    print("=" * 80)
    print(f"Query: {QUERY}\n")

    # --------------------------------------------------------------
    # Step 1: Initialize Retriever
    # --------------------------------------------------------------
    retriever = DenseRetriever()

    # --------------------------------------------------------------
    # Step 2: Index sample documents
    # (Minimal setup – no dataset dependency)
    # --------------------------------------------------------------
    documents = [
        {
            "id": "doc1",
            "text": (
                "Diabetes is a chronic disease that affects how the body "
                "processes blood sugar. Common symptoms include fatigue, "
                "excessive thirst, and frequent urination."
            ),
        },
        {
            "id": "doc2",
            "text": (
                "Treatment for diabetes often involves lifestyle changes "
                "such as diet and exercise. In some cases, medication or "
                "insulin therapy is required."
            ),
        },
        {
            "id": "doc3",
            "text": (
                "If left untreated, diabetes can lead to serious complications "
                "including heart disease, kidney failure, and vision problems."
            ),
        },
    ]

    retriever.index_documents(documents)
    print(f"✓ Indexed {len(documents)} documents\n")

    # --------------------------------------------------------------
    # Step 3: Retrieve
    # --------------------------------------------------------------
    retrieved_docs = retriever.retrieve(QUERY, top_k=TOP_K)

    original_texts = [doc["text"] for doc, _ in retrieved_docs]
    original_context = " ".join(original_texts)

    print("Original Context:")
    print(original_context)
    print("\nOriginal Token Count:", approx_tokens(original_context))
    print("-" * 80)

    # --------------------------------------------------------------
    # Step 4: EXIT Compression
    # --------------------------------------------------------------
    exit_compressor = ExitBaselineCompressor(token=None)

    start = time.time()
    exit_compressed_docs = [
        exit_compressor.compress(QUERY, text)
        for text in original_texts
    ]
    exit_time = time.time() - start

    exit_context = " ".join(exit_compressed_docs)

    print("EXIT Compressed Context:")
    print(exit_context)
    print("\nEXIT Token Count:", approx_tokens(exit_context))
    print(f"EXIT Time: {exit_time:.2f}s")
    print("-" * 80)

    # --------------------------------------------------------------
    # Step 5: EP-EXIT Compression
    # --------------------------------------------------------------
    ep_exit = EPExitCompressor(window_size=2)

    start = time.time()
    ep_contexts = []
    for text in original_texts:
        sentences = ep_exit.split_into_sentences(text)
        units = ep_exit.build_evidence_units(sentences)
        scored = ep_exit.score_units(QUERY, units)
        selected = ep_exit.select_top_units(
            scored, max_tokens=EXIT_TOKEN_BUDGET
        )
        ep_contexts.append(ep_exit.reconstruct_context(selected))

    ep_time = time.time() - start
    ep_exit_context = " ".join(ep_contexts)

    print("EP-EXIT Compressed Context:")
    print(ep_exit_context)
    print("\nEP-EXIT Token Count:", approx_tokens(ep_exit_context))
    print(f"EP-EXIT Time: {ep_time:.2f}s")
    print("-" * 80)

    # --------------------------------------------------------------
    # Step 6: Summary
    # --------------------------------------------------------------
    results = {
        "query": QUERY,
        "original_tokens": approx_tokens(original_context),
        "exit_tokens": approx_tokens(exit_context),
        "ep_exit_tokens": approx_tokens(ep_exit_context),
        "exit_time": exit_time,
        "ep_exit_time": ep_time,
        "exit_context": exit_context,
        "ep_exit_context": ep_exit_context,
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/exit_vs_ep_exit.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("✓ Results saved to results/exit_vs_ep_exit.json")
    print("=" * 80)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
