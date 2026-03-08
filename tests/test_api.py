import requests
import time
import sys
import random

BASE_URL = "http://localhost:8000"

def print_header(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def test_api_flow():
    try:
        requests.get(f"{BASE_URL}/docs")
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to {BASE_URL}. Is app.py running?")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 1: Get Datasets and Pick Random Query
    # ---------------------------------------------------------
    print_header("1. Fetching Datasets (GET /datasets)")
    response = requests.get(f"{BASE_URL}/datasets")
    assert response.status_code == 200, f"Failed to fetch datasets: {response.text}"
    
    datasets = response.json()
    print("✓ Successfully fetched datasets.")
    
    # Select HotpotQA if available to test the dynamic indexing, else fallback to JWST
    dataset_id = "hotpotqa" if "hotpotqa" in datasets and datasets["hotpotqa"]["queries"] else "jwst"
    
    # Randomly select a query
    sample_query = random.choice(datasets[dataset_id]["queries"])
    print(f"Selected Dataset: {dataset_id}")
    print(f"Randomly Selected Query: '{sample_query}'")

    # ---------------------------------------------------------
    # Test 2: Execute Auto-Indexed Query
    # ---------------------------------------------------------
    print_header("2. Executing Auto-Indexed Query (POST /query)")
    query_payload = {
        "query": sample_query,
        "top_k": 4,
        "compare_original": True,
        "use_coarse": True,
        "use_fine": True
    }
    
    print("Waiting for auto-indexing and pipeline processing...")
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/query", json=query_payload)
    elapsed = time.time() - start_time
    
    assert response.status_code == 200, f"Query failed: {response.text}"
    result = response.json()
    
    print(f"✓ Query & Auto-indexing returned successfully in {elapsed:.2f}s!")
    
    # ---------------------------------------------------------
    # Test 3: Validate
    # ---------------------------------------------------------
    print_header("3. Validating API Response Payload")
    expected_keys = ['query', 'answer', 'original_docs_answer', 'retrieved_docs', 'metrics', 'ep_exit_details', 'quitox_details']
    for key in expected_keys:
        assert key in result, f"Missing expected key: {key}"
        print(f"  ✓ Found '{key}'")
        
    print("\n-- Answer Comparison --")
    print(f"Compressed Answer: {result['answer']}")
    print(f"Original Answer:   {result['original_docs_answer']}")
    
    print("\n-- Key Metrics --")
    m = result['metrics']
    print(f"Compression Ratio: {m['compression']['ratio_chars']:.1f}%")
    print(f"Token Savings:     {m['compression']['ratio_tokens']:.1f}%")
    
    print("\n✅ API tests passed!")

    # ---------------------------------------------------------
    # Test 4: Detailed Pipeline Diagnostics
    # ---------------------------------------------------------
    print_header("4. Detailed Pipeline Diagnostics")
    print(f"Target Query: {result['query']}\n")

    print("--- [RETRIEVAL PHASE] ---")
    for doc in result['retrieved_docs']:
        # Print document index, score, and a short snippet of the text
        print(f"📄 Doc {doc['doc_index']} | Score: {doc['score']:.4f}")
        snippet = doc['text'][:120].replace('\n', ' ') + "..." if len(doc['text']) > 120 else doc['text']
        print(f"   Snippet: {snippet}\n")

    print("--- [COMPRESSION PHASE (EP-EXIT)] ---")
    for detail in result['ep_exit_details']:
        doc_idx = detail.get('doc_index')
        print(f"\n📑 Document {doc_idx} Compression Decisions:")
        
        kept = detail.get('kept_units', [])
        removed = detail.get('removed_units', [])

        print(f"  ✓ KEPT ({len(kept)} sentences):")
        for unit in kept:
            # Handle different dictionary structures safely
            text = unit.get('text', str(unit)) if isinstance(unit, dict) else str(unit)
            score = unit.get('score', 0.0) if isinstance(unit, dict) else 0.0
            print(f"    [{score:.2f}] {text}")

        print(f"  ✗ REMOVED ({len(removed)} sentences):")
        for unit in removed:
            text = unit.get('text', str(unit)) if isinstance(unit, dict) else str(unit)
            score = unit.get('score', 0.0) if isinstance(unit, dict) else 0.0
            print(f"    [{score:.2f}] {text}")

if __name__ == "__main__":
    test_api_flow()