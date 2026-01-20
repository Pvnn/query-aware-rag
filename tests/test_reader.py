import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generation.reader import RAGReader

def test_reader():
  print("Test: Answer Generation")
  
  reader = RAGReader()
  
  query = "How do SSDs work?"
  context = "SSDs use flash memory to store data. They have faster read/write speeds than HDDs."
  
  print(f"Query: {query}")
  print(f"Context: {context}\n")
  
  answer = reader.generate_answer(query, context)
  
  print(f"Generated Answer: {answer}")
  
  assert len(answer) > 0
  print("\n✓ Reader test passed")

if __name__ == "__main__":
  test_reader()
