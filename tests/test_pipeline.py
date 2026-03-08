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
  
  documents = [
    "The James Webb Space Telescope (JWST) is a space telescope designed to conduct infrared astronomy. Its high-resolution and high-sensitivity instruments allow it to view objects too old, distant, or faint for the Hubble Space Telescope. The primary scientific instruments on board include the Near-Infrared Camera (NIRCam) and the Mid-Infrared Instrument (MIRI). These tools are essential for studying the formation of early galaxies. Additionally, the telescope requires a massive sunshield to keep its instruments cold.",
    "The Hubble Space Telescope was launched into low Earth orbit in 1990 and remains in operation. It features a 2.4-meter mirror and observes primarily in the visible and ultraviolet spectra. Hubble has recorded some of the most detailed visible-light images ever, allowing a deep view into space. It does not possess the advanced mid-infrared capabilities of newer observatories. Astronauts have visited Hubble multiple times to repair and upgrade its systems.",
    "JWST was launched on December 25, 2021, on an Ariane 5 rocket from Kourou, French Guiana. It took a month to reach its destination, the Sun-Earth L2 Lagrange point. This orbit allows the telescope to stay in line with Earth as it moves around the Sun. The L2 point is approximately 1.5 million kilometers away from our planet. Operations and data processing are managed by the Space Telescope Science Institute.",
    "One of the major goals of modern astronomy is the study of exoplanets and their atmospheres. JWST uses its Near-Infrared Spectrograph (NIRSpec) to analyze the chemical composition of exoplanetary atmospheres. By observing the transit of a planet across its host star, scientists can detect signatures of water and carbon dioxide. This method of transit spectroscopy is highly dependent on the telescope's stable orbit. Future missions will continue this search for habitable worlds."
  ]
  
  pipeline = QueryAwareRAG(token=token)
  pipeline.retriever.index_documents(documents)
  
  query = "What are the specific instruments on the James Webb Space Telescope used for observation?"
  result = pipeline.run(query, top_k=4, compare_original=True)
  
  # --- RIGOROUS KEY MAPPING ---
  m = result['metrics']
  times = m['times']
  comp = m['compression']
  usage = m['usage']

  print("\n" + "="*60)
  print("✓ Pipeline test passed")
  print("="*60)
  
  print("\n📊 Overall Metrics:")
  print(f"  - Total Pipeline Time: {times['total_pipeline']:.2f}s")
  if result.get('original_docs_answer'):
    print(f"  - Original Path Time:  {times['original_total']:.2f}s")
    print(f"  - Net Time Saved:      {times['net_time_saved']:.2f}s ({'FASTIER' if times['net_time_saved'] > 0 else 'SLOWER'})")
  
  print(f"\n📉 Compression Efficiency:")
  print(f"  - Payload Ratio (Chars): {comp['ratio_chars']:.1f}%")
  print(f"  - Token Savings (Lit):   {comp['ratio_tokens']:.1f}%")
  print(f"  - Local Compute Cost:    {comp['hf_tokens_cost']} HF tokens")

  print(f"\n💳 API Usage & Cost:")
  print(f"  - Compressed API Total:  {usage['compressed_api_tokens']} tokens")
  if result.get('original_docs_answer'):
    print(f"  - Original API Total:    {usage['original_api_tokens']} tokens")

  # --- QUITO-X COARSE LOGS ---
  if result.get('quitox_details'):
    print("\n🎯 QUITO-X Stage (Coarse Sentence Filtering):")
    for doc_log in result['quitox_details']:
      print(f"\n  Document {doc_log['doc_index']}:")
      for entry in doc_log['details']:
        status = "✅" if entry['retained'] else "❌"
        print(f"    {status} [{entry['score']:.4f}] {entry['text']}")

  # --- EP-EXIT FINE LOGS ---
  if result.get('ep_exit_details'):
    print("\n🔍 EP-EXIT Stage (Fine Evidence Extraction):")
    for doc_details in result['ep_exit_details']:
      doc_idx = doc_details['doc_index']
      print(f"\n  Document {doc_idx}:")
      print(f"    Units Created: {doc_details['evidence_units_total']} | Kept: {doc_details['evidence_units_kept_count']}")
      
      if doc_details.get('kept_units'):
        print("    ✅ KEPT:")
        for u in doc_details['kept_units']:
          print(f"       [{u['score']:.2f}] {u['text']}")
          
      if doc_details.get('removed_units'):
        print("    ❌ REMOVED:")
        for u in doc_details['removed_units']:
          print(f"       [{u['score']:.2f}] {u['text']}")

  # --- ANSWER COMPARISON SECTION ---
  print("\n" + "  ANSWER COMPARISON ")
  print("-" * 60)
  
  if result.get('original_docs_answer'):
    print("ORIGINAL DOCS ANSWER (Uncompressed):")
    print(f"   {result['original_docs_answer']}")
    print("\nCOMPRESSED PIPELINE ANSWER:")
    print(f"   {result['answer']}")
  else:
    print(f"✅ PIPELINE ANSWER: {result['answer']}")

  print("-" * 60)
  
  if result.get('original_docs_answer'):
    orig_len = len(result['original_docs_answer'].split())
    comp_len = len(result['answer'].split())
    print(f"Info: Original answer length: {orig_len} words | Compressed answer length: {comp_len} words")

if __name__ == "__main__":
  test_pipeline()