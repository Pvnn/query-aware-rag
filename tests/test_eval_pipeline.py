import sys
import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eval.eval_pipeline import GenerativeEvaluator
from src.eval.interfaces import SearchResult
from src.eval.metrics import answer_em_correctness

from src.compression.hybrid_compressor import HybridCompressor
from src.generation.reader import RAGReader

class HybridAdapter:
    """Wraps the HybridCompressor to match the GenerativeEvaluator interface."""
    def __init__(self, compressor):
        self.compressor = compressor

    def compress(self, query, docs):
        full_text = " ".join([d.text for d in docs])
        result = self.compressor.compress(
            query=query, 
            context=full_text, 
            coarse_ratio=0.7, 
            fine_threshold=0.5, 
            use_coarse=True, 
            use_fine=True
        )
        return {
            "compressed_docs": [
                SearchResult(evi_id=0, docid=0, title="hybrid", text=result['final_text'])
            ]
        }

def test_real_pipeline_eval():
    print("Loading Environment and Models...")

    load_dotenv()
    token = os.getenv("HF_TOKEN")

    # Initialize models
    compressor = HybridCompressor(exit_token=token)
    adapter = HybridAdapter(compressor)
    
    # Using Ollama Reader
    reader = RAGReader(model_name="llama3.1:8b")

    evaluator = GenerativeEvaluator(compressor=adapter, reader=reader)

    # Dummy dataset matching HotpotQA format
    dummy_data = [
        {
            "question": "What are the specific instruments on the James Webb Space Telescope used for observation?",
            "answer": "Near-Infrared Camera", # This is the exact string HotpotQA expects
            "context": [
                [
                    "JWST",
                    [
                        "The James Webb Space Telescope (JWST) is a space telescope designed to conduct infrared astronomy.",
                        "Its high-resolution and high-sensitivity instruments allow it to view objects too old, distant, or faint for the Hubble Space Telescope.",
                        "The primary scientific instruments on board include the Near-Infrared Camera (NIRCam) and the Mid-Infrared Instrument (MIRI).",
                        "These tools are essential for studying the formation of early galaxies.",
                        "Additionally, the telescope requires a massive sunshield to keep its instruments cold.",
                    ],
                ]
            ]
        }
    ]

    print("\n" + "="*50)
    print(" 1. DIAGNOSTIC TRACE (Debugging EM)")
    print("="*50)
    
    query = dummy_data[0]["question"]
    gt_answer = dummy_data[0]["answer"]
    
    # 1. Manually compress
    full_text = " ".join(dummy_data[0]["context"][0][1])
    docs = [SearchResult(evi_id=0, docid=0, title="JWST", text=full_text)]
    comp_result = adapter.compress(query, docs)
    compressed_text = comp_result["compressed_docs"][0].text
    
    # 2. Manually generate
    gen_result = reader.generate_answer(query, compressed_text, strict_mode=True)
    generated_text = gen_result["answer"]
    
    # 3. Test EM function manually
    em_score = answer_em_correctness(generated_text, [gt_answer], ignore_case=True)
    
    print(f"QUESTION:      {query}")
    print(f"GROUND TRUTH:  '{gt_answer}'")
    print(f"LLM OUTPUT:    '{generated_text}'")
    print("-" * 50)
    print(f"IS ANSWER IN OUTPUT? : {bool(em_score)} (Score: {em_score})")
    print("-" * 50)


    print("\n" + "="*50)
    print(" 2. RUNNING GENERATIVE PIPELINE EVALUATOR")
    print("="*50)

    results = evaluator.evaluate(dummy_data)

    print("\n📊 Generative Evaluation Results:")
    print(f"Exact Match (EM):      {results['em']:.1f}%")
    print(f"ROUGE-L:               {results['rougeL']:.1f}%")
    print(f"Disambig-F1:           {results['disambig_f1']:.1f}%")
    print(f"Compression Ratio:     {results['compression_ratio_chars']:.1f}%")
    print(f"Avg Latency:           {results['avg_latency_sec']:.2f}s")


if __name__ == "__main__":
    test_real_pipeline_eval()