import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def print_header(title):
  print(f"\n{'='*50}")
  print(f" {title}")
  print(f"{'='*50}")

def test_api_flow():
  # Ensure server is running
  try:
    requests.get(f"{BASE_URL}/docs")
  except requests.exceptions.ConnectionError:
    print(f"❌ Error: Could not connect to {BASE_URL}. Is app.py running?")
    sys.exit(1)

  # Test 1: Get Available Datasets & Queries
  print_header("1. Fetching Datasets (GET /datasets)")
  response = requests.get(f"{BASE_URL}/datasets")
  assert response.status_code == 200, f"Failed to fetch datasets: {response.text}"
  
  datasets = response.json()
  print("✓ Successfully fetched datasets:")
  print(json.dumps(datasets, indent=2))
  
  sample_query = datasets["jwst"]["queries"][0]

  # Test 2: Load the JWST Dataset
  print_header("2. Loading JWST Corpus (POST /dataset/load)")
  payload = {"dataset_id": "jwst"}
  response = requests.post(f"{BASE_URL}/dataset/load", json=payload)
  assert response.status_code == 200, f"Failed to load dataset: {response.text}"
  
  print("✓ Successfully loaded corpus:")
  print(json.dumps(response.json(), indent=2))

  # Test 3: Execute RAG Pipeline with Compression
  print_header("3. Executing Query (POST /query)")
  query_payload = {
    "query": sample_query,
    "top_k": 4,
    "compare_original": True,
    "use_coarse": True,
    "use_fine": True
  }
  
  print(f"Sending Query: '{sample_query}'...")
  print("Waiting for pipeline to process (this may take a few seconds)...")
  
  start_time = time.time()
  response = requests.post(f"{BASE_URL}/query", json=query_payload)
  elapsed = time.time() - start_time
  
  assert response.status_code == 200, f"Query failed: {response.text}"
  result = response.json()
  
  print(f"✓ Query returned successfully in {elapsed:.2f}s!")
  
  # Test 4: Validate Frontend Contract (JSON Structure)
  print_header("4. Validating API Response Payload")
  
  # Verify core keys exist
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
  print(f"Time Saved:        {m['times']['net_time_saved']:.2f}s")
  
  print("\n-- EP-EXIT Visual Data (First Doc) --")
  if result['ep_exit_details']:
    first_doc_details = result['ep_exit_details'][0]
    print(f"Doc Index: {first_doc_details.get('doc_index')}")
    print(f"Sentences Kept: {len(first_doc_details.get('kept_units', []))}")
    print(f"Sentences Dropped: {len(first_doc_details.get('removed_units', []))}")
  
  print("\n✅ All API tests passed!")

if __name__ == "__main__":
  test_api_flow()