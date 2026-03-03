import os
import sys
from pathlib import Path

# Make src importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.compression.ep_exit import EPExitCompressor


def test_ep_exit():
    print("=== Testing EP-EXIT ===\n")

    # Get HF token from environment
    token = os.getenv("HF_TOKEN")

    if token is None:
        raise ValueError("HF_TOKEN not set in environment.")

    # Sample document
    document = """
    Insulin is a hormone produced by the pancreas.
    It helps regulate blood sugar levels in the body.
    The Eiffel Tower is located in Paris.
    Diabetes occurs when the body cannot produce enough insulin.
    Insulin therapy is commonly used to treat diabetes.
    Paris is the capital of France.
    """

    query = "How does insulin work in diabetes?"

    compressor = EPExitCompressor(token=token)

    compressed = compressor.compress(query, document)

    print("Original Document:\n")
    print(document)

    print("\nCompressed Output:\n")
    print(compressed)

    print("\n✓ EP-EXIT test completed")


if __name__ == "__main__":
    test_ep_exit()