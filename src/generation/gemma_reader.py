import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

class GemmaRAGReader:
  def __init__(self, model_name="gemma-3-27b-it"):
    print(f"Initializing RAGReader with GenAI SDK (Model: {model_name})...")
    self.model_name = model_name
    
    # Load environment variables from the .env file
    load_dotenv()
    self.api_key = os.getenv("GEMINI_API_KEY")
    
    if not self.api_key:
      raise ValueError(
        "GEMINI_API_KEY not found. Please ensure you have a .env file "
        "with GEMINI_API_KEY=your_key_here"
      )
      
    # Initialize the new SDK Client
    try:
      self.client = genai.Client(api_key=self.api_key)
      
      # A lightweight test to ensure the API key is valid and model is accessible
      test_response = self.client.models.generate_content(
        model=self.model_name,
        contents="test",
        config=types.GenerateContentConfig(max_output_tokens=1)
      )
      
      if test_response.text:
        print(f"Using Google GenAI backend with {self.model_name}\n")
      else:
        raise RuntimeError("Received an empty response during initialization test.")
        
    except Exception as e:
      error_msg = str(e)
      if "API_KEY_INVALID" in error_msg:
        raise RuntimeError("Invalid API key provided. Please check your .env file.")
      elif "not found" in error_msg.lower():
        raise RuntimeError(f"Model '{model_name}' not found or you don't have access to it.")
      else:
        raise RuntimeError(f"GenAI API error during setup: {error_msg}")

  def generate_answer(self, query, context, max_new_tokens=100):
    """
    Generate answer given query and context using the new google.genai SDK.
    
    Args:
      query: Question string
      context: Context string (compressed documents)
      max_new_tokens: Maximum tokens to generate
      
    Returns:
      Dictionary containing the generated answer and token usage statistics.
    """
    # Strict prompt to enforce context-only answering
    prompt = f"""You are a helpful assistant that answers questions ONLY using the provided context. Do not use any external knowledge.

Context: {context}

Question: {query}

Answer (using only the context above):"""
    
    try:
      response = self.client.models.generate_content(
        model=self.model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
          temperature=0.0,            # 0 ensures deterministic, focused answers
          max_output_tokens=max_new_tokens,
          top_p=1.0
        )
      )
      
      # Default usage tracking in case it's blocked
      usage_stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
      }
      
      # Extract exact token usage from the API response
      if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage_stats = {
          "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0) or 0,
          "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0) or 0,
          "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0) or 0
        }
      
      # Handle potential empty responses from safety filters
      if not response.text:
        return {
          "answer": "Error: Output was blocked by safety filters or returned empty.",
          "usage": usage_stats
        }
        
      return {
        "answer": response.text.strip(),
        "usage": usage_stats
      }
      
    except Exception as e:
      raise RuntimeError(f"GenAI API error during generation: {str(e)}")