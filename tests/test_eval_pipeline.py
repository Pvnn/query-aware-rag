import sys
import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eval.eval_pipeline import CompressorEvaluator
from src.compression.hybrid_compressor import HybridCompressor


def test_real_pipeline_eval():
    print("Loading Environment and Models...")

    load_dotenv()
    token = os.getenv("HF_TOKEN")

    # Initialize compressor directly (NO adapter)
    compressor = HybridCompressor(exit_token=token)

    evaluator = CompressorEvaluator(compressor=compressor)

    # Dummy dataset matching HotpotQA format
    dummy_data = [
        {
            "question": "What are the specific instruments on the James Webb Space Telescope used for observation?",
            "answer": "Near-Infrared Camera",
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
            ],
            "supporting_facts": [
                ["JWST", 1],
                ["JWST", 2],
            ],
        }
    ]

    print("\n--- Running Extractive Evaluation ---")

    results = evaluator.evaluate(dummy_data)

    print("\n📊 Extractive Evaluation Results:")
    print(f"Context Precision:    {results['context_precision']:.1f}%")
    print(f"Context Recall:       {results['context_recall']:.1f}%")
    print(f"Context F1:           {results['context_f1']:.1f}%")
    print(f"Answer Survival Rate: {results['answer_survival_rate']:.1f}%")
    print(f"Compression Ratio:    {results['compression_ratio_chars']:.1f}%")
    print(f"Avg Latency:          {results['avg_latency_sec']:.2f}s")


if __name__ == "__main__":
    test_real_pipeline_eval()