import requests

class RAGReader:
  def __init__(self, model_name="llama3.1:8b"):
    print(f"Initializing RAGReader with Ollama {model_name}...")
    self.model_name = model_name
    self.ollama_url = "http://localhost:11434/api/generate"
    
    # Test if Ollama is running and model exists
    try:
      test_response = requests.post(self.ollama_url, 
        json={
          'model': self.model_name,
          'prompt': 'test',
          'stream': False
        },
        timeout=10
      )
      if test_response.status_code == 200:
        print(f"Using Ollama backend with {model_name}\n")
      else:
        error_data = test_response.json()
        error_msg = error_data.get('error', 'Unknown error')
        if 'not found' in error_msg.lower():
          raise RuntimeError(
            f"Model '{model_name}' not found. Install it with:\n"
            f"  ollama pull {model_name}"
          )
        raise RuntimeError(f"Ollama error: {error_msg}")
    except requests.exceptions.ConnectionError:
      raise RuntimeError("Ollama is not running.")
    except requests.exceptions.Timeout:
      raise RuntimeError("Ollama took too long to respond. Is it running?")
  
  def generate_answer(self, query, context, max_new_tokens=100):
    """
    Generate answer given query and context.
    
    Args:
      query: Question string
      context: Context string (compressed documents)
      max_new_tokens: Maximum tokens to generate
      
    Returns:
      Generated answer string
    """
    # Stricter prompt to enforce context-only answering
    prompt = f"""You are a helpful assistant that answers questions ONLY using the provided context. Do not use any external knowledge.

  Context: {context}

  Question: {query}

  Answer (using only the context above):"""
    
    try:
      response = requests.post(self.ollama_url,
        json={
          'model': self.model_name,
          'prompt': prompt,
          'stream': False,
          'options': {
            'temperature': 0,
            'num_predict': max_new_tokens,
            'top_p': 1.0
          }
        },
        timeout=60
      )
      
      if response.status_code != 200:
        error_data = response.json()
        error_msg = error_data.get('error', 'Unknown error')
        raise RuntimeError(f"Ollama API error: {error_msg}")
      
      result = response.json()
      return result['response'].strip()
      
    except requests.exceptions.Timeout:
      raise RuntimeError("Generation timed out (>60s)")
    except KeyError as e:
      raise RuntimeError(f"Unexpected response format: {response.text}")

