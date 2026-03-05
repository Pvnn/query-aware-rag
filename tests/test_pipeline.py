import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
from src.rag_pipeline import QueryAwareRAG

def test_pipeline():
  print("Test: Complete RAG Pipeline with Complex Corpus")
  
  # Sample corpus designed to test semantic grouping and fine-grained filtering
  documents = [
    # Doc 1: Highly relevant, contains specific answers mixed with general context
    "The James Webb Space Telescope (JWST) is a space telescope designed to conduct infrared astronomy. Its high-resolution and high-sensitivity instruments allow it to view objects too old, distant, or faint for the Hubble Space Telescope. The primary scientific instruments on board include the Near-Infrared Camera (NIRCam) and the Mid-Infrared Instrument (MIRI). These tools are essential for studying the formation of early galaxies. Additionally, the telescope requires a massive sunshield to keep its instruments cold.",
    
    # Doc 2: Related domain, but completely irrelevant to the specific query (Hubble instead of JWST)
    "The Hubble Space Telescope was launched into low Earth orbit in 1990 and remains in operation. It features a 2.4-meter mirror and observes primarily in the visible and ultraviolet spectra. Hubble has recorded some of the most detailed visible-light images ever, allowing a deep view into space. It does not possess the advanced mid-infrared capabilities of newer observatories. Astronauts have visited Hubble multiple times to repair and upgrade its systems.",
    
    # Doc 3: Same subject (JWST), but irrelevant to the query (Launch and Orbit details)
    "JWST was launched on December 25, 2021, on an Ariane 5 rocket from Kourou, French Guiana. It took a month to reach its destination, the Sun-Earth L2 Lagrange point. This orbit allows the telescope to stay in line with Earth as it moves around the Sun. The L2 point is approximately 1.5 million kilometers away from our planet. Operations and data processing are managed by the Space Telescope Science Institute.",
    
    # Doc 4: Highly relevant, mentions another instrument and its specific use case
    "One of the major goals of modern astronomy is the study of exoplanets and their atmospheres. JWST uses its Near-Infrared Spectrograph (NIRSpec) to analyze the chemical composition of exoplanetary atmospheres. By observing the transit of a planet across its host star, scientists can detect signatures of water and carbon dioxide. This method of transit spectroscopy is highly dependent on the telescope's stable orbit. Future missions will continue this search for habitable worlds."
  ]
  
  # Initialize pipeline
  pipeline = QueryAwareRAG(token=token)
  
  # Index documents
  pipeline.retriever.index_documents(documents)
  
  # Run query focusing on a specific aspect to force aggressive filtering
  query = "What are the specific instruments on the James Webb Space Telescope used for observation?"
  result = pipeline.run(query, top_k=4)
  
  # Validate
  assert len(result['answer']) > 0
  assert result['metrics']['total_time'] > 0
  
  print("\n" + "="*60)
  print("✓ Pipeline test passed")
  print("="*60)
  
  print("\n📊 Overall Metrics:")
  print(f"  - Total time: {result['metrics']['total_time']:.2f}s")
  print(f"  - Compression: {result['metrics']['compression_ratio']:.1f}%")
  print(f"  - Tokens: {result['metrics']['original_tokens']} → {result['metrics']['compressed_tokens']}")
  
  # Deep dive into the EP-EXIT stats to see the evidence units in action
  print("\n🔍 EP-EXIT Deep Dive (Evidence Units):")
  for doc_details in result['ep_exit_details']:
    doc_idx = doc_details['doc_index']
    print(f"\n  Document {doc_idx}:")
    print(f"    Total Units Created: {doc_details['evidence_units_total']}")
    print(f"    Units Kept: {doc_details['evidence_units_kept_count']}")
    print(f"    Units Removed: {doc_details['evidence_units_removed_count']}")
    
    if doc_details['kept_units']:
      print("    ✅ KEPT Text (Passed Threshold):")
      for u in doc_details['kept_units']:
        print(f"       [{u['score']:.2f}] {u['text']}")
        
    if doc_details['removed_units']:
      print("    ❌ REMOVED Text (Failed Threshold):")
      for u in doc_details['removed_units']:
        print(f"       [{u['score']:.2f}] {u['text']}")

if __name__ == "__main__":
  test_pipeline()