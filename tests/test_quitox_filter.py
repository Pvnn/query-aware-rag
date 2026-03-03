import unittest
from src.compression.quitox_filter import QuitoxCoarseFilter


class TestQuitoxRealIntegration(unittest.TestCase):
  """
  Integration tests using the REAL model and REAL text.
  """

  @classmethod
  def setUpClass(cls):
    """Load the model once for all tests in this class to save time."""
    print("\nLoading QUITO-X model for integration testing...")
    cls.filter = QuitoxCoarseFilter(model_name="google/flan-t5-small")
    print("✅ Model loaded.")

  def test_sanity_check_filtering(self):
    """
    Give a clear query and a mix of obvious relevant/irrelevant sentences.
    The model should prioritize the relevant ones.
    """
    query = "What is the capital of France?"

    # A mix of sentences
    sentences = [
      "The sky is blue and the grass is green.",  # Irrelevant (Nature)
      "Paris is the capital and most populous city of France.",  # Highly Relevant
      "Bananas are a popular fruit worldwide.",  # Irrelevant (Food)
      "The Eiffel Tower is located in Paris, France.",  # Somewhat Relevant (Context)
      "Python is a programming language.",  # Irrelevant (Tech)
    ]

    # We want to keep top 40% (2 out of 5 sentences)
    # We expect the "Paris" sentence and maybe "Eiffel Tower" to be kept.
    compression_ratio = 0.4

    filtered = self.filter.compress(query, sentences, compression_ratio)

    print(f"\n🔎 Query: {query}")
    print(f"👉 Input ({len(sentences)}): {sentences}")
    print(f"👉 Output ({len(filtered)}): {filtered}")

    # assertions
    self.assertEqual(len(filtered), 2, "Should return exactly 2 sentences")

    # The most obvious sentence MUST be there
    self.assertIn(
      "Paris is the capital and most populous city of France.",
      filtered,
      "Failed to keep the most relevant sentence!",
    )

    # The most obvious irrelevant sentence should NOT be there
    self.assertNotIn(
      "Bananas are a popular fruit worldwide.",
      filtered,
      "Kept an obviously irrelevant sentence!",
    )


if __name__ == "__main__":
  unittest.main()
